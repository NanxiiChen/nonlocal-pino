"""Isolated horizon/thickness controls with common weighted local evaluation.

Training uses only prescribed IC/BC and PDE residuals. The radial local fields
are a separately qualified far-field surrogate of the cube problem and are
loaded solely for evaluation. No native nonlocal solution is assumed here.
"""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.icemelting3d.extreme_reference import weighted_metrics
from examples.icemelting3d.thin_interface import build_network, crossing, make_loss_fn, sample_batch
from examples.icemelting3d.thin_interface_operators import compact_stencil, offsets_and_weights
from examples.icemelting3d.thin_interface_reference import parameters
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy


def validate(c):
    if (c.experiment.kind != 'extreme_interface' or c.nl.method not in ('sparse', 'local')
            or c.phase.formulation != 'ic_latent' or c.arch.type != 'mlp'
            or c.arch.use_hard_constraint or c.arch.out_dim != 1
            or c.sample.Nx % 2
            or tuple(c.nondim.xc) != (.5, .5, .5)
            or c.experiment.normalization != 'divide_by_N3'):
        raise ValueError('Extreme study requires the centered scalar hard-IC latent MLP and explicit derivative mode')
    counts = (c.train.iterations, c.sample.K_core, c.sample.Nt,
        c.experiment.boundary_points, c.experiment.evaluation_chunk, c.log.eval_every)
    if any(type(v) is not int or v <= 0 for v in counts):
        raise ValueError('Positive integer training, sampling and evaluation counts required')
    values = (*parameters(c), c.experiment.boundary_weight)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Positive finite physical coefficients and boundary weight required')
    return offsets_and_weights(c)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                            for x in jax.tree.leaves(tree) if eqx.is_array(x)]))


def predictor(c, times):
    @eqx.filter_jit
    def predict(net, points):
        u = jax.vmap(lambda t: jax.vmap(net.latent, in_axes=(0, None))(points, t)[:, 0])(
            jnp.asarray(times, dtype=jnp.float32))
        return jnp.tanh(u), jnp.all(jnp.isfinite(u))

    def evaluate(net, points):
        result = np.empty((len(times), len(points)), dtype=np.float64)
        chunk = c.experiment.evaluation_chunk
        for start in range(0, len(points), chunk):
            xx = points[start:start+chunk]
            count = len(xx)
            xx = np.pad(xx, ((0, chunk-count), (0, 0)), mode='edge')
            values, finite = predict(net, jnp.asarray(xx, dtype=jnp.float32))
            if not bool(finite):
                raise FloatingPointError('Nonfinite latent field in checkpoint evaluation')
            result[:, start:start+count] = np.asarray(values)[:, :count]
        return result
    return evaluate


def load_reference(c):
    path = Path(c.experiment.reference_path)
    saved = json.loads((path.parent/'config.json').read_text())
    current = json.loads(c.to_json_best_effort())
    for key in ('physical', 'nondim', 'reference'):
        if saved[key] != current[key]:
            raise ValueError(f'Extreme reference {key} mismatch: {path}')
    for key in ('snapshots', 'quadrature_directions', 'quadrature_radial',
                'radial_points', 'radial_directions'):
        if saved['experiment'][key] != current['experiment'][key]:
            raise ValueError(f'Extreme reference evaluation {key} mismatch: {path}')
    report = json.loads((path.parent/'validation.json').read_text())
    if report['status'] != 'passed' or report['reference_kind'] != 'radial_latent_far_field_surrogate':
        raise ValueError('Extreme study requires an explicitly qualified radial reference')
    with np.load(path, allow_pickle=False) as data:
        reference = {key: data[key] for key in data.files}
    if (not all(np.isfinite(value).all() for value in reference.values())
            or reference['points'].shape != (len(reference['weights']), 3)
            or reference['local'].shape != (c.experiment.snapshots, len(reference['weights']))
            or not np.all(reference['weights'] > 0)
            or np.any((reference['points'] < 0) | (reference['points'] > 1))
            or not np.array_equal(reference['time'], np.linspace(0, 1, c.experiment.snapshots))
            or reference['rays'].shape != (c.experiment.radial_directions, c.experiment.radial_points, 3)
            or reference['local_rays'].shape != (c.experiment.snapshots, c.experiment.radial_directions, c.experiment.radial_points)
            or reference['radius'].shape != (c.experiment.radial_points,)
            or not np.all(np.diff(reference['radius']) > 0)):
        raise ValueError('Invalid weighted reference fields/coordinates')
    return reference


def train(c):
    operator = validate(c)
    ref = load_reference(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    run = Path(c.log.log_dir)/c.log.run_name
    run.mkdir(parents=True, exist_ok=False)
    (run/'config.json').write_text(c.to_json_best_effort()+'\n')
    _write(run/'operator.json', operator[2])
    net = build_network(c)
    opt = make_optimizer(c)
    state = opt.init(eqx.filter(net, eqx.is_array))
    eqx.tree_serialise_leaves(run/'model_initial.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_initial.eqx', state)
    loss_fn = make_loss_fn(c)

    @eqx.filter_jit
    def step(net, state, points, sub, times, boundary, normal):
        (loss, parts), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            net, points, sub, times, boundary, normal)
        updates, next_state = opt.update(grad, state, net)
        candidate = eqx.apply_updates(net, updates)
        norm = optax.global_norm(grad)
        flags = jnp.stack((jnp.isfinite(loss), _finite(grad), _finite(candidate),
                           _finite(next_state), jnp.isfinite(norm)))
        return candidate, next_state, loss, parts, flags, norm

    predict = predictor(c, ref['time'])
    update_times, preparation_times = [], []
    trace = dict(ids=[], times=[], boundary=[], normal=[])
    failure, completed = None, 0
    started = time.perf_counter()
    with (run/'loss.jsonl').open('x') as log:
        for i in range(c.train.iterations):
            start = time.perf_counter()
            ids, times, boundary, normal = sample_batch(c, i)
            for key, value in zip(trace, (ids, times, boundary, normal), strict=True):
                trace[key].append(value)
            if c.nl.method == 'local':
                points, sub = ids.astype(np.float32)/(c.sample.Nx-1), None
            else:
                points, sub = compact_stencil(ids, c, operator)
            payload = jax.device_put((points, sub, times, boundary, normal))
            jax.block_until_ready(payload)
            preparation_times.append(time.perf_counter()-start)
            start = time.perf_counter()
            candidate, next_state, loss, parts, flags, gradient_norm = step(net, state, *payload)
            jax.block_until_ready((candidate, next_state, loss, flags))
            update_times.append(time.perf_counter()-start)
            if not bool(jnp.all(flags)):
                failure = dict(attempted_step=i+1, finite_flags=np.asarray(flags).tolist(),
                    flag_order=['loss', 'gradient', 'model', 'optimizer', 'gradient_norm'],
                    loss=str(float(loss)), gradient_norm=str(float(gradient_norm)),
                    checkpoint='last finite model; failed update was not installed')
                _write(run/'numerical-failure.json', failure)
                print('NUMERICAL_FAILURE', json.dumps(failure), flush=True)
                break
            net, state = candidate, next_state
            completed = i+1
            if completed % 100 == 0:
                norm = float(gradient_norm)
                record = dict(step=completed, loss=float(loss), parts=np.asarray(parts).tolist(),
                    gradient_norm=norm if np.isfinite(norm) else str(norm),
                    clipped=norm > c.train.grad_clip)
                log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                print('TRAIN', json.dumps(record), flush=True)
            if completed % c.log.eval_every == 0 or completed == c.train.iterations:
                pred = predict(net, ref['points'])
                record = dict(step=completed, **weighted_metrics(pred, ref['local'], ref['weights']))
                with (run/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                eqx.tree_serialise_leaves(run/f'model_{completed:06d}.eqx', net)
                print('EVAL', json.dumps(record), flush=True)
    eqx.tree_serialise_leaves(run/'model_final.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_final.eqx', state)
    np.savez_compressed(run/'sampling.npz', **{key: np.stack(value) for key, value in trace.items()})
    np.savez_compressed(run/'timing.npz', update=update_times, preparation=preparation_times)
    if failure is not None or completed % c.log.eval_every:
        pred = predict(net, ref['points'])
    rays = predict(net, ref['rays'].reshape(-1, 3)).reshape(ref['local_rays'].shape)
    np.savez_compressed(run/'prediction.npz', phi=pred, rays=rays)
    metrics = weighted_metrics(pred, ref['local'], ref['weights'])
    counts = np.sum((rays[1:, :, :-1] >= 0) & (rays[1:, :, 1:] < 0), axis=-1)
    topology = dict(missing=int(np.sum(counts == 0)), multiple=int(np.sum(counts > 1)), total=int(counts.size))
    metrics['radius_relative_rmse'] = None
    if not topology['missing']:
        r = crossing(rays, ref['radius'])[1:]
        target = crossing(ref['local_rays'], ref['radius'])[1:]
        metrics['radius_relative_rmse'] = float(np.sqrt(np.mean(((r-target)/target)**2)))
    warm = np.asarray(update_times[20:])+np.asarray(preparation_times[20:])
    summary = dict(status='complete' if failure is None else 'numerical_failure',
        requested_steps=c.train.iterations, completed_steps=completed, seed=c.train.seed,
        epsilon=c.physical.eps_star, method=c.nl.method, operator=operator[2],
        metrics=metrics, topology=topology, failure=failure, cold_start=True,
        median_step_seconds=float(np.median(warm)) if len(warm) else None,
        first_step_seconds=update_times[0], elapsed_seconds=time.perf_counter()-started,
        gpu=str(jax.devices()[0].device_kind), memory=jax.devices()[0].memory_stats(),
        reference='Independently qualified radial local far-field surrogate; common cube quadrature',
        native_reference_included=False, final_model='model_final.eqx',
        saved_model_step=completed, final_optimizer='optimizer_final.eqx')
    _write(run/'summary.json', summary)
    print('COMPLETE', json.dumps(summary, allow_nan=False), flush=True)
