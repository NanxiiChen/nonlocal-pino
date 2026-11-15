"""Isolated full-workload Taylor-AD resource probe with exact original inputs.

This diagnostic does not change the production trainer or its dispatch. Its
gradient accumulation and SOAP update follow operator_local_ad.make_step;
only the already-qualified spatial derivative implementation changes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from docs.tasks.spinodal_taylor_ch_qualification import candidate_parts
from examples.confined_ch2d.operator_precision import compute_arrays, compute_dtype

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def make_step(c, optimizer):
    from examples.confined_ch2d.operator_local_ad import validate_local_ad
    validate_local_ad(c)

    def objective(net, data, times):
        parts = candidate_parts(net, data, times, c)
        return parts[0]+c.operator.local_ad.boundary_weight*jnp.sum(parts[1:]), parts

    @eqx.filter_jit
    def step(net, state, batch, times):
        batch, times = compute_arrays(batch, c), jnp.asarray(times, dtype=compute_dtype(c))
        count = len(batch['points'])
        gradient0 = jax.tree.map(jnp.zeros_like, eqx.filter(net, eqx.is_array))

        def add(i, totals):
            value, gradient, parts = totals
            data = jax.tree.map(lambda a: a[i], batch)
            (vi, pi), gi = eqx.filter_value_and_grad(objective, has_aux=True)(net, data, times)
            return value+vi/count, jax.tree.map(lambda a, b: a+b/count, gradient, gi), parts+pi/count

        value, gradient, parts = jax.lax.fori_loop(0, count, add,
            (jnp.asarray(0., dtype=compute_dtype(c)), gradient0, jnp.zeros(3, dtype=compute_dtype(c))))
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]))
        updates, state = optimizer.update(gradient, state, net)
        return eqx.apply_updates(net, updates), state, value, finite, parts
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    args = parser.parse_args()
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU must be idle before isolated resource probe: '+occupied)
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    if uuid != args.expected_gpu_uuid:
        raise RuntimeError('The comparison GPU identity changed')
    if shutil.disk_usage(args.run.parent).free < 8*1024**3:
        raise RuntimeError('Less than8GiBdisk headroom; no launch')
    c = ConfigDict(json.loads((args.baseline/'config.json').read_text()))
    if c.train.iterations != 30 or c.train.compute_dtype != 'float32':
        raise ValueError('Require the existing cold30 FP32 baseline configuration')
    qualification = json.loads((ROOT/'artifacts/spinodal_taylor_ch_production_fp32_qualification.json').read_text())
    if qualification['status'] != 'passed' or qualification['architecture'] != json.loads(c.arch.to_json()):
        raise ValueError('Production architecture parameter-backward qualification missing/mismatched')
    from examples.confined_ch2d.operator_comparison import configure_comparison
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    configure_comparison(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Full resource probe requires the declared GPU')
    args.run.mkdir(parents=True, exist_ok=False)
    c.log.run_name = args.run.name
    write(args.run/'config.json', c.to_dict())
    status = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  baseline=str(args.baseline), gpu_uuid=uuid, method='local_ad_taylor', completed_updates=0)
    write(args.run/'status.json', status)
    pool = None
    try:
        names = json.loads((args.baseline/'source/files.json').read_text())
        for name in names:
            if (args.baseline/'source'/name).read_bytes() != (ROOT/name).read_bytes():
                raise ValueError('Baseline numerical source changed: '+name)
        names += ['docs/tasks/spinodal_taylor_ad_probe.py',
                  'docs/tasks/spinodal_taylor_ch_qualification.py',
                  'docs/tasks/spinodal_taylor_resource_probe.py']
        for name in names:
            target = args.run/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        write(args.run/'source/files.json', names)
        from examples.confined_ch2d.operator_training import load_bank, build_network, operator_temporal_samples
        from examples.confined_ch2d.operator_precision import make_compute_optimizer
        from examples.confined_ch2d.operator_online import RollingGeometryPool
        from examples.confined_ch2d.operator_continuation import initial_state
        from examples.confined_ch2d.operator_problem import evaluation_indices
        start = time.perf_counter()
        manifest, refs, bank = load_bank(c)
        keys = jax.random.split(jax.random.key(c.train.seed), 3)
        net, optimizer = build_network(c, keys[0]), make_compute_optimizer(c)
        net, state, start_step, _ = initial_state(net, optimizer, None)
        if start_step != 0:
            raise AssertionError('Cold initialization required')
        for name, tree in (('model_initial.eqx', net), ('optimizer_initial.eqx', state)):
            eqx.tree_serialise_leaves(args.run/name, tree)
            if (args.run/name).read_bytes() != (args.baseline/name).read_bytes():
                raise AssertionError('Initial state differs from the original probe: '+name)
        _, labeled = evaluation_indices(manifest)
        ids = np.asarray(labeled, dtype=np.int32)
        known_keys = ('points', 'volumes', 'initial', 'condition', 'condition_weights')
        np.savez_compressed(args.run/'known_encoder_inputs.npz', indices=ids,
                            **{key: np.asarray(bank[key][ids]) for key in known_keys})
        with np.load(args.run/'known_encoder_inputs.npz') as actual, np.load(args.baseline/'known_encoder_inputs.npz') as old:
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], old[key])
        with np.load(args.baseline/'training_draws.npz') as old:
            expected = {k: np.array(old[k]) for k in old.files}
        step = make_step(c, optimizer)
        pool = RollingGeometryPool(c, args.run, 0, None,
                                   [refs[index]['holes'] for index in manifest['test_indices']])
        bank = pool.start(bank)
        timings, preparations, draws, times_draws, geometry_draws = [], [], [], [], []
        setup = time.perf_counter()-start
        for i in range(1, 31):
            bank = pool.refresh(bank, i)
            ids = jax.random.randint(jax.random.fold_in(keys[1], i),
                                    (c.operator.condition_batch,), 0, c.operator.train_conditions)
            tick = time.perf_counter()
            batch = jax.tree.map(lambda a: a[ids], bank)
            preparations.append(time.perf_counter()-tick)
            times, _ = operator_temporal_samples(c, i, jax.random.fold_in(keys[2], i))
            geom_ids = pool.draw_ids[np.asarray(ids)].copy()
            np.testing.assert_array_equal(ids, expected['condition_ids'][i-1])
            np.testing.assert_array_equal(times, expected['times'][i-1])
            np.testing.assert_array_equal(geom_ids, expected['geometry_draw_ids'][i-1])
            tick = time.perf_counter()
            net, state, value, finite, parts = step(net, state, batch, times)
            value.block_until_ready()
            timings.append(time.perf_counter()-tick)
            if not bool(finite):
                raise FloatingPointError(f'Nonfinite loss/parameter gradient at update{i}')
            draws.append(np.asarray(ids)); times_draws.append(np.asarray(times)); geometry_draws.append(geom_ids)
            row = dict(step=i, loss=float(value), parts=np.asarray(parts).tolist(), update_seconds=timings[-1])
            with (args.run/'loss.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            status.update(completed_updates=i, heartbeat_utc=datetime.now(timezone.utc).isoformat())
            write(args.run/'status.json', status)
            if i in (1, 25, 30):
                print(json.dumps(row), flush=True)
            if i == 25:
                leaves = [np.asarray(a) for a in jax.tree.leaves((net, state)) if eqx.is_inexact_array(a)]
                if not all(np.isfinite(a).all() and a.dtype == compute_dtype(c) for a in leaves):
                    raise FloatingPointError('Full FP32 model/optimizer preflight failed')
                write(args.run/'online_preflight.json', dict(status='passed', step=25,
                    all_floating_model_state_finite_and_compute_dtype=True))
        with np.load(args.run/'first_online_encoder_inputs.npz') as actual, np.load(args.baseline/'first_online_encoder_inputs.npz') as old:
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], old[key])
        np.savez_compressed(args.run/'training_draws.npz', condition_ids=draws,
                            geometry_draw_ids=geometry_draws, times=times_draws)
        np.savez_compressed(args.run/'update_timing.npz', update_seconds=timings, batch_preparation_seconds=preparations)
        memory = jax.devices()[0].memory_stats()
        median = float(np.median((np.asarray(timings)+preparations)[10:]))
        summary = dict(status='passed', updates=30, warm_inclusive_median_seconds=median,
            projected50k_update_hours=median*50000/3600, within_execution_ceiling=median*50000/3600 <= 24,
            setup_seconds=setup, first_update_seconds=timings[0], internal_wall_seconds=time.perf_counter()-start,
            peak_active_gib=memory['peak_bytes_in_use']/1024**3, device_memory_stats=memory,
            allocator_headroom_gate=memory['peak_bytes_in_use'] < .95*memory['bytes_limit'],
            exact_original_initial_model_optimizer_and_input_draws=True, pool=pool.summary(),
            scope='Full production architecture/points/B16/Nt16 and SOAP; no reference evaluation, only resource and finite-state qualification',
            limitations='Reduced startup assessment; cold-process cost is not directly comparable to baseline training. No final accuracy result.')
        write(args.run/'summary.json', summary)
        status.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        write(args.run/'status.json', status)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.run/'status.json', status)
        raise
    finally:
        if pool is not None:
            pool.close()


if __name__ == '__main__':
    main()
