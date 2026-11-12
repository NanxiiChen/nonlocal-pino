"""Wait for the 16-run melting queue, refresh analysis, and audit 128^3 metrics.

Run as a module with explicit --queue and --timeout-seconds. The timeout bounds
the wait for training; no numerical backend is imported before that wait ends.
Original analysis files are backed up, and no dense fields are written.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time


WIDTHS = (.0225, .015, .010, .0075)
SEEDS = (42, 43)
METHODS = ('sparse', 'local')
REPLAY_ATOL = 1e-5
REPLAY_RTOL = 0.


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def owner_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def training_budget(journal):
    """Validate new queue schedules while preserving the recorded 5k schema."""
    steps = journal['steps']
    if type(steps) is not int or steps <= 0:
        raise ValueError('Queue steps must be a positive integer')
    if 'lr_decay_every' in journal:
        decay_every = journal['lr_decay_every']
        if type(decay_every) is not int or decay_every <= 0:
            raise ValueError('Queue lr_decay_every must be a positive integer')
        return steps, decay_every
    if steps != 5000:
        raise ValueError('Only the legacy 5000-update queue may omit lr_decay_every')
    # The legacy queue used the unchanged study configuration's declared decay.
    return steps, None


def wait_for_queue(queue, timeout_seconds, update):
    deadline = time.monotonic()+timeout_seconds
    expected = {(eps, seed, method) for eps in WIDTHS for seed in SEEDS for method in METHODS}
    while True:
        journal = read_json(queue/'journal.json')
        training_budget(journal)
        if tuple(journal['widths']) != WIDTHS or tuple(journal['seeds']) != SEEDS:
            raise ValueError('Expected the declared four widths and seeds42/43')
        runs = [stage for stage in journal['stages'] if stage['kind'] == 'train']
        identities = [(r['epsilon'], r['seed'], r['method']) for r in runs]
        if len(set(identities)) != len(identities) or not set(identities) <= expected:
            raise ValueError('Unexpected or duplicate training identity in queue')
        passed = [r for r in runs if 'status' in r and r['status'] == 'passed']
        if any(r['exit_code'] != 0 for r in passed):
            raise ValueError('Passed training stage has nonzero exit code')
        ready = len(passed) == 16
        terminal = journal['status'] in ('complete', 'failed')
        alive = owner_alive(journal['pid'])
        update('waiting', completed_training=len(passed), queue_status=journal['status'])
        if ready and (terminal or not alive):
            # Passed is published only after Popen.wait() reaps each GPU child.
            return journal, runs
        if terminal or not alive or any('status' in r and r['status'] == 'failed' for r in runs):
            raise RuntimeError('Training queue stopped before all16training stages passed')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Queue wait exceeded {timeout_seconds}seconds')
        time.sleep(min(30., remaining))


def validate_runs(queue, journal, runs):
    from docs.tasks.melting_thin_interface_study import config, tag
    steps, decay_every = training_budget(journal)
    checked = []
    for run in runs:
        folder = Path(run['run'])
        saved = read_json(folder/'config.json')
        expected = json.loads(config(run['epsilon']).to_json_best_effort())
        expected['train']['seed'] = run['seed']
        expected['train']['iterations'] = steps
        if decay_every is not None:
            expected['train']['lr_decay_every'] = decay_every
        expected['nl']['method'] = run['method']
        reference = Path(journal['reference_root'])/tag(run['epsilon'])/'reference.npz'
        expected['experiment']['reference_path'] = str(reference)
        for key in ('physical', 'nondim', 'arch', 'phase', 'nl', 'sample', 'train', 'loss', 'experiment'):
            if saved[key] != expected[key]:
                raise ValueError(f'Saved numerical configuration mismatch: {folder}: {key}')
        if (not folder.is_absolute() or not reference.is_absolute()
                or (Path(saved['log']['log_dir'])/saved['log']['run_name']).resolve() != folder.resolve()):
            raise ValueError(f'Invalid direct run/reference path: {folder}')
        summary = read_json(folder/'summary.json')
        for key in ('epsilon', 'seed', 'method'):
            if summary[key] != run[key]:
                raise ValueError(f'Run summary identity mismatch: {folder}: {key}')
        if summary['steps'] != steps or summary['final_model'] != 'model_final.eqx':
            raise ValueError(f'Expected final{steps}checkpoint: {folder}')
        for name in ('model_final.eqx', 'prediction.npz', 'sampling.npz'):
            if not (folder/name).is_file():
                raise FileNotFoundError(folder/name)
        reference_config = read_json(reference.parent/'config.json')
        for key in ('physical', 'nondim', 'sample'):
            if reference_config[key] != saved[key]:
                raise ValueError(f'Reference configuration mismatch: {reference}: {key}')
        for key in ('delta', 'kernel'):
            if reference_config['nl'][key] != saved['nl'][key]:
                raise ValueError(f'Reference graph mismatch: {reference}: {key}')
        for key, value in saved['experiment'].items():
            if key != 'reference_path' and reference_config['experiment'][key] != value:
                raise ValueError(f'Reference numerical setting mismatch: {reference}: {key}')
        checked.append((folder, saved, summary, reference))
    return checked


def topology(rays):
    import numpy as np
    if rays.ndim != 3 or not np.isfinite(rays).all():
        raise ValueError('Expected finite(time,direction,radius)ray fields')
    counts = np.sum((rays[1:, :, :-1] >= 0) & (rays[1:, :, 1:] < 0), axis=-1)
    return dict(positive_time=True, total=int(counts.size), missing=int(np.sum(counts == 0)),
                multiple=int(np.sum(counts > 1)), outward_crossings_by_time_direction=counts.tolist())


def check_replay(replay, saved):
    """Check physical FP32 phase values, including the sharply varying hard IC."""
    import numpy as np
    if (replay.shape != saved.shape or replay.ndim != 2 or replay.shape[0] < 2
            or not np.isfinite(replay).all() or not np.isfinite(saved).all()):
        raise ValueError('Replay requires matching finite(time,point)fields with positive times')
    difference = np.abs(replay-saved)
    report = dict(replay_max_abs=float(difference.max()),
        replay_initial_max_abs=float(difference[0].max()),
        replay_positive_time_max_abs=float(difference[1:].max()),
        replay_atol=REPLAY_ATOL, replay_rtol=REPLAY_RTOL)
    # u0=(R-r)/(sqrt(2)*epsilon) amplifies FP32 radius-evaluation roundoff.
    # Use an absolute physical-phase tolerance; near zero relative error is ill-conditioned.
    np.testing.assert_allclose(replay, saved, atol=REPLAY_ATOL, rtol=REPLAY_RTOL,
                               err_msg=json.dumps(report))
    return report


def predictor(c, net, times):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    @eqx.filter_jit
    def predict(current, points):
        return jax.vmap(lambda t: jax.vmap(current, in_axes=(0, None))(points, t)[:, 0])(
            jnp.asarray(times, dtype=jnp.float32))
    def evaluate(points):
        result = np.empty((len(times), len(points)), dtype=np.float64)
        chunk = c.experiment.evaluation_chunk
        for start in range(0, len(points), chunk):
            xx = points[start:start+chunk]
            count = len(xx)
            xx = np.pad(xx, ((0, chunk-count), (0, 0)), mode='edge')
            values = np.asarray(predict(net, jnp.asarray(xx, dtype=jnp.float32)))
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise FloatingPointError('Checkpoint inference must return finiteFP32values')
            result[:, start:start+count] = values[:, :count]
        return result
    return evaluate


def finish(queue, checked, update):
    import equinox as eqx
    import jax
    import numpy as np
    from ml_collections import ConfigDict
    from docs.tasks.melting_thin_interface_study import analyze
    from examples.icemelting3d.thin_interface import build_network, field_metrics, validate
    from examples.icemelting3d.thin_interface_reference import evaluation_points, sample_reference
    from nonlocal_pinn.training.runtime import configure_gpu_policy

    backup = queue/'postprocess-original-analysis'
    backup.mkdir(exist_ok=False)
    for name in ('results.json', 'error-vs-thickness.png', 'position-and-time.png', 'convergence.png'):
        if (queue/name).exists():
            shutil.copy2(queue/name, backup/name)
    update('analysis', original_analysis_backup=str(backup))
    analyze(queue)
    receipt = dict(queue=str(queue), original_analysis_backup=str(backup), topology=[], quadrature=[],
        reference_dtype='float64', inference_dtype='float32', dense_fields_written=False,
        replay_atol=REPLAY_ATOL, replay_rtol=REPLAY_RTOL,
        metric_scope='Positive-time quadrature comparison with unchanged local FD512 discretization',
        qualification='128^3 coincides with the training spatial support; this is not an off-grid holdout')
    for folder, saved, _, reference in checked:
        with np.load(folder/'prediction.npz', allow_pickle=False) as prediction:
            receipt['topology'].append(dict(run=str(folder), epsilon=saved['physical']['eps_star'],
                seed=saved['train']['seed'], method=saved['nl']['method'], **topology(prediction['rays'])))
    receipt_path = queue/'postprocess-receipt.json'
    write_json(receipt_path, receipt)
    thin = [item for item in checked if item[1]['physical']['eps_star'] == min(WIDTHS)]
    if len(thin) != 4:
        raise ValueError('Expected all four thinnest final checkpoints')
    c = ConfigDict(thin[0][1])
    validate(c)
    configure_gpu_policy(c.train.gpu_policy)
    c.experiment.evaluation_axis = 128
    update('reference128', reference_grid=512, evaluation_axis=128)
    with jax.enable_x64(True):
        points, _, _ = evaluation_points(c)
        sampled, seconds = sample_reference(c, 512, c.experiment.reference_dt, local=True)
    truth = sampled[:, :len(points)]
    if truth.dtype != np.float64 or not np.isfinite(truth).all():
        raise FloatingPointError('New reference must contain finiteFP64fields')
    receipt['reference'] = dict(source_config=str(thin[0][3].parent/'config.json'),
        representation=c.experiment.reference_form, solver_grid=512, evaluation_axis=128,
        dt=c.experiment.reference_dt, seconds=seconds, latent_snapshot_finiteness_checked=True)
    for folder, saved, summary, reference in thin:
        update('inference128', run=str(folder))
        with jax.enable_x64(False):
            current = ConfigDict(saved)
            jax.config.update('jax_default_matmul_precision', current.train.matmul_precision)
            net = eqx.tree_deserialise_leaves(folder/'model_final.eqx', build_network(current))
            if any(np.issubdtype(leaf.dtype, np.inexact) and leaf.dtype != np.float32
                   for leaf in jax.tree.leaves(net) if eqx.is_array(leaf)):
                raise ValueError(f'Checkpoint has non-FP32inexact leaves: {folder}')
            with np.load(reference, allow_pickle=False) as original:
                np.testing.assert_array_equal(original['time'], np.linspace(0, 1, c.experiment.snapshots))
                evaluate = predictor(current, net, original['time'])
                replay = evaluate(original['points'])
                with np.load(folder/'prediction.npz', allow_pickle=False) as prediction:
                    replay_report = check_replay(replay, prediction['phi'])
                    original_metrics = field_metrics(prediction['phi'], original['local'])
            del replay
            dense_metrics = field_metrics(evaluate(points), truth)
        for key, value in original_metrics.items():
            np.testing.assert_allclose(value, summary['metrics'][key], rtol=1e-12, atol=1e-14)
        receipt['quadrature'].append(dict(run=str(folder), checkpoint=str(folder/'model_final.eqx'),
            seed=saved['train']['seed'], method=saved['nl']['method'], epsilon=min(WIDTHS),
            original_axis=64, dense_axis=128, **replay_report,
            original_metrics=original_metrics, dense_metrics=dense_metrics,
            change_percentage_points={key: 100*(dense_metrics[key]-value)
                                      for key, value in original_metrics.items()}))
        write_json(receipt_path, receipt)
    return receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True, type=Path)
    parser.add_argument('--timeout-seconds', required=True, type=float)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be finite and positive')
    queue = args.queue.resolve(strict=True)
    state = dict(status='starting', pid=os.getpid(), queue=str(queue),
                 started_utc=datetime.now(timezone.utc).isoformat(), timeout_seconds=args.timeout_seconds)
    journal_path = queue/'postprocess-journal.json'
    with journal_path.open('x') as stream:
        stream.write(json.dumps(state, indent=2)+'\n')
    def update(status, **values):
        state.update(status=status, **values)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(journal_path, state)
        print('POSTPROCESS', json.dumps(state), flush=True)
    try:
        journal, runs = wait_for_queue(queue, args.timeout_seconds, update)
        checked = validate_runs(queue, journal, runs)
        receipt = finish(queue, checked, update)
        update('complete', receipt=str(receipt))
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
