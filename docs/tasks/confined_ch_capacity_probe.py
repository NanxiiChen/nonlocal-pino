"""Explicit capacity feasibility and fresh-initialization gates; no training labels."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import re
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from nonlocal_pinn.training.runtime import configure_gpu_policy, with_run_seed
from docs.tasks.confined_ch_capacity_cases import capacity_case
from examples.confined_ch2d.operator_problem import validate
from examples.confined_ch2d.operator_precision import make_compute_optimizer
from examples.confined_ch2d.operator_training import (
    build_network, fields, load_bank, make_loss, operator_temporal_samples,
)
from docs.tasks.confined_ch_fp32_probe import check32, cpu_replay


def config_pair(prior, case='width192'):
    spec = capacity_case(case)
    c = with_run_seed(importlib.import_module(spec['config']).get_config(), None)
    old = ConfigDict(json.loads((prior / 'config.json').read_text()))
    expected = json.loads(old.to_json())
    expected['arch']['width'] = spec['width']
    expected['arch']['spatial_features'] = spec['features']
    expected['train']['soap_max_precond_dim'] = 512
    expected['train']['iterations'] = spec['iterations']
    expected['log']['run_name'] = c.log.run_name
    if spec['control'] == 'bank_growth':
        expected['operator']['train_conditions'] = 1024
        expected['operator']['dataset_path'] = '/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1'
        expected['operator']['precompute']['source_path'] = old.operator.dataset_path
        expected['operator']['precompute']['reuse_mode'] = 'hardlink'
    if expected != json.loads(c.to_json()):
        raise AssertionError('Configuration differs beyond the declared capacity/bank/budget control')
    if (old.arch.width, old.arch.spatial_features, old.arch.heads,
            old.train.soap_max_precond_dim, old.train.compute_dtype) != (
                spec['baseline_width'], spec['baseline_features'], 4, spec['baseline_cap'], 'float32'):
        raise ValueError('Capacity baseline differs from the explicitly declared completed FP32 control')
    validate(c)
    return c, old


def configuration_audit(prior, output, case='width192'):
    if output.exists():
        raise FileExistsError(output)
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Configuration audit must use CPU')
    c, old = config_pair(prior, case)
    spec = capacity_case(case)
    records = []
    for config in (old, c):
        net = eqx.filter_eval_shape(lambda: build_network(config, jax.random.key(config.train.seed)))
        count = sum(a.size for a in jax.tree.leaves(net) if isinstance(a, jax.ShapeDtypeStruct)) - net.frequencies.size
        skipped = []
        for path, a in jax.tree_util.tree_leaves_with_path(net):
            if isinstance(a, jax.ShapeDtypeStruct) and a.ndim > 1 and jax.tree_util.keystr(path) != '.frequencies':
                for axis, size in enumerate(a.shape):
                    if size > config.train.soap_max_precond_dim:
                        skipped.append(dict(path=jax.tree_util.keystr(path), shape=list(a.shape), axis=axis))
        parameters = eqx.filter(net, lambda a: isinstance(a, jax.ShapeDtypeStruct))
        state = jax.eval_shape(make_compute_optimizer(config).init, parameters)
        model_bytes = sum(a.size * a.dtype.itemsize for a in jax.tree.leaves(parameters))
        state_bytes = sum(math.prod(a.shape) * a.dtype.itemsize for a in jax.tree.leaves(state)
                          if isinstance(a, jax.ShapeDtypeStruct))
        records.append(dict(width=config.arch.width, head_dimension=config.arch.width // config.arch.heads,
            spatial_sin_cos_features=2 * config.arch.spatial_features, trainable_parameters=count,
            soap_axis_cap=config.train.soap_max_precond_dim, skipped_matrix_axes=skipped,
            model_bytes=model_bytes, optimizer_state_bytes=state_bytes))
        if config is old and spec['baseline_cap'] == 256:
            assert not any(isinstance(a, jax.ShapeDtypeStruct) and any(256 < n <= 512 for n in a.shape)
                           for path, a in jax.tree_util.tree_leaves_with_path(net)
                           if jax.tree_util.keystr(path) != '.frequencies')
    assert [r['trainable_parameters'] for r in records] == [spec['baseline_parameters'], spec['parameters']]
    assert [len(r['skipped_matrix_axes']) for r in records] == [1, 1]
    keys = jax.random.split(jax.random.key(c.train.seed), 3)
    expected_draws = np.asarray([jax.random.randint(jax.random.fold_in(keys[1], i),
        (c.operator.condition_batch,), 0, c.operator.train_conditions) for i in range(1, 26)])
    if spec['control'] == 'capacity':
        with np.load(prior / 'training_draws.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(expected_draws, f['condition_ids'][:25])
    else:
        template = build_network(c, jax.random.key(0))
        previous = eqx.tree_deserialise_leaves(prior / 'model_initial.eqx', template)
        fresh = build_network(c, keys[0])
        for a, b in zip(jax.tree.leaves(previous), jax.tree.leaves(fresh), strict=True):
            if eqx.is_array(a):
                np.testing.assert_array_equal(a, b)
        if not np.any(expected_draws >= old.operator.train_conditions):
            raise AssertionError('Expanded sampling prefix does not visit new training geometries')
    for i in [*range(1, 26), 10000, 30000, spec['iterations']]:
        key = jax.random.fold_in(keys[2], i)
        a, _ = operator_temporal_samples(c, i, key)
        b, _ = operator_temporal_samples(old, i, key)
        np.testing.assert_array_equal(a, b)
    # Keep every paired intermediate model/SOAP state plus three spare pairs,
    # per-pair serialization overhead, probe/media budget and 1GiB free margin.
    pairs = c.train.iterations // c.log.checkpoint_every + 3
    disk_bound = pairs * (records[-1]['model_bytes'] + records[-1]['optimizer_state_bytes'] + 8 * 2**20) + 2**30
    result = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
        prior=str(prior), case=case, controls=records,
        baseline_cap256_and512_coverage_identical=spec['baseline_cap'] == 256,
        paired_checkpoint_count_with_spares=pairs, required_free_disk_bytes=max(4 * 2**30, disk_bound),
        budget=c.train.iterations,
        expected_geometry_prefix_matches_completed_baseline=spec['control'] == 'capacity',
        exact_initial_arrays_match_completed_baseline=spec['control'] == 'bank_growth',
        geometry_sampling_scope='Same global keys; expanded support intentionally changes condition IDs'
            if spec['control'] == 'bank_growth' else 'Identical condition IDs to completed baseline',
        time_samples_identical=True,
        scope='CPU configuration/abstract shape audit; no larger-network timing, memory or accuracy measured')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def probe(prior, output, case='width192'):
    output.mkdir(parents=True, exist_ok=False)
    c, old = config_pair(prior, case)
    spec = capacity_case(case)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Capacity probe requires exclusive GPU')
    (output / 'config.json').write_text(c.to_json(indent=2) + '\n')
    started = time.perf_counter()
    print(f'Loading full{c.operator.train_conditions + c.operator.test_conditions}geometry bank for B16/Nt16 width{c.arch.width} FP32 probe', flush=True)
    manifest, _, bank = load_bank(c)
    assert set(bank) == {'points', 'volumes', 'initial', 'neighbors', 'coefficients', 'metric', 'condition', 'condition_weights'}
    check32(bank)
    keys = jax.random.split(jax.random.key(c.train.seed), 3)
    net = build_network(c, keys[0])
    if spec['control'] == 'bank_growth':
        saved = eqx.tree_deserialise_leaves(prior / 'model_initial.eqx', build_network(c, jax.random.key(0)))
        for a, b in zip(jax.tree.leaves(net), jax.tree.leaves(saved), strict=True):
            if eqx.is_array(a):
                np.testing.assert_array_equal(a, b)
        del saved
    eqx.tree_serialise_leaves(output / 'model_initial.eqx', net)
    opt = make_compute_optimizer(c)
    state = opt.init(eqx.filter(net, eqx.is_array))
    loss = make_loss(c)

    @eqx.filter_jit
    def step(model, state, batch, times):
        value, gradient = eqx.filter_value_and_grad(loss)(model, batch, times)
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]))
        update, state = opt.update(gradient, state, model)
        return eqx.apply_updates(model, update), state, value, finite

    ids = jax.random.randint(jax.random.fold_in(keys[1], 1), (c.operator.condition_batch,), 0, c.operator.train_conditions)
    batch = jax.tree.map(lambda a: a[ids], bank)
    times, _ = operator_temporal_samples(c, 1, jax.random.fold_in(keys[2], 1))
    lower = step.lower(net, state, batch, times)
    ir = lower.as_text()
    (output / 'step.stablehlo').write_text(ir)
    promoted = re.findall(r'tensor<[^>]*xf64>', ir)
    if promoted:
        raise AssertionError(f'Non-scalar FP64 in capacity training IR: {sorted(set(promoted))}')
    tick = time.perf_counter()
    compiled = lower.compile()
    compile_seconds = time.perf_counter() - tick
    durations, losses, draws = [], [], []
    for iteration in range(1, 26):
        ids = jax.random.randint(jax.random.fold_in(keys[1], iteration),
            (c.operator.condition_batch,), 0, c.operator.train_conditions)
        batch = jax.tree.map(lambda a: a[ids], bank)
        key = jax.random.fold_in(keys[2], iteration)
        times, _ = operator_temporal_samples(c, iteration, key)
        old_times, _ = operator_temporal_samples(old, iteration, key)
        np.testing.assert_array_equal(times, old_times)
        tick = time.perf_counter()
        net, state, value, finite = compiled(net, state, batch, times)
        value.block_until_ready()
        durations.append(time.perf_counter() - tick)
        if not bool(finite):
            raise FloatingPointError(f'Capacity probe nonfinite gradient at{iteration}')
        check32((net, state))
        losses.append(float(value))
        draws.append(np.asarray(ids))
        print(json.dumps(dict(step=iteration, seconds=durations[-1], loss=losses[-1])), flush=True)
    expected_draws = np.asarray([jax.random.randint(jax.random.fold_in(keys[1], i),
        (c.operator.condition_batch,), 0, c.operator.train_conditions) for i in range(1, 26)])
    np.testing.assert_array_equal(draws, expected_draws)
    if spec['control'] == 'capacity':
        with np.load(prior / 'training_draws.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(draws, f['condition_ids'][:25])
    draw_map = np.asarray([record['draw_index'] for record in manifest['conditions']])
    if np.isin(draw_map[np.asarray(draws)], c.operator.precompute.reserved_test_draws).any():
        raise AssertionError('Reserved geometry sampled by optimizer')
    validation_index = manifest['test_indices'][0]
    data = jax.tree.map(lambda a: a[validation_index], bank)
    times = jnp.array([0., .75, 1.5, 2.25, 3.], dtype=jnp.float32)
    prediction = np.asarray(eqx.filter_jit(lambda n, d: fields(n, d, times, c))(net, data))
    check32(prediction)
    eqx.tree_serialise_leaves(output / 'model_probe.eqx', net)
    np.savez_compressed(output / 'cpu_replay.npz', **{k: np.asarray(v) for k, v in data.items()},
        times=np.asarray(times), prediction=prediction)
    memory = jax.devices()[0].memory_stats()
    if memory['peak_bytes_in_use'] > .95 * memory['bytes_limit']:
        raise RuntimeError('Capacity probe has insufficient allocator headroom')
    result = dict(status='passed', prior=str(prior), case=case, steps=25, trainable_parameters=spec['parameters'],
        width=c.arch.width, heads=c.arch.heads, head_dimension=c.arch.width // c.arch.heads,
        spatial_sin_cos_features=2 * c.arch.spatial_features, soap_axis_cap=c.train.soap_max_precond_dim,
        actual_geometry_prefix_matches_baseline=spec['control'] == 'capacity',
        actual_geometry_draws_match_global_seed=True, no_reserved_geometry_draws=True,
        validation_index=validation_index, time_draws_identical=True,
        all_compute_leaves_fp32=True, all_losses_gradients_states_finite=True,
        stablehlo_nonscalar_fp64_tensors=0, stablehlo_fp64_scalar_tokens=len(re.findall(r'tensor<f64>', ir)),
        compile_seconds=compile_seconds, first_execution_seconds=durations[0],
        warm_seconds=float(np.mean(durations[5:])), durations=durations, losses=losses,
        memory=memory, wall_seconds=time.perf_counter() - started,
        scope='25discarded full-bank feasibility steps; exact same fresh192initial arrays, expanded geometry support, no trained weights imported.'
            if spec['control'] == 'bank_growth' else
            '25discarded full-bank feasibility steps. Same initial seed/sampler, different-shaped fresh parameters; no pretrained weights or claimed L2 gain.')
    (output / 'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('durations', 'losses')}, indent=2), flush=True)


def initial_audit(prior, run, probe_path, output, case='width192'):
    if output.exists():
        raise FileExistsError(output)
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Initial audit must use CPU')
    expected_config, _ = config_pair(prior, case)
    c = ConfigDict(json.loads((run / 'config.json').read_text()))
    normalized = json.loads(c.to_json())
    normalized['log']['run_name'] = expected_config.log.run_name
    assert normalized == json.loads(expected_config.to_json())
    template = build_network(c, jax.random.key(0))
    saved = eqx.tree_deserialise_leaves(run / 'model_initial.eqx', template)
    preflight = eqx.tree_deserialise_leaves(probe_path / 'model_initial.eqx', template)
    fresh = build_network(c, jax.random.split(jax.random.key(c.train.seed), 3)[0])
    count = 0
    for a, b, d in zip(jax.tree.leaves(saved), jax.tree.leaves(preflight), jax.tree.leaves(fresh), strict=True):
        if eqx.is_array(a):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, d)
            count += 1
    if capacity_case(case)['control'] == 'bank_growth':
        baseline = eqx.tree_deserialise_leaves(prior / 'model_initial.eqx', template)
        for a, b in zip(jax.tree.leaves(saved), jax.tree.leaves(baseline), strict=True):
            if eqx.is_array(a):
                np.testing.assert_array_equal(a, b)
    for source in ('src/nonlocal_pinn/nn/point_conditional.py', 'src/nonlocal_pinn/training/optimizers.py',
            'src/nonlocal_pinn/training/runtime.py', 'examples/confined_ch2d/operator_training.py',
            'examples/confined_ch2d/operator_precision.py'):
        assert (run / 'source' / source).read_bytes() == (prior / 'source' / source).read_bytes(), source
    result = dict(status='passed', run=str(run), prior=str(prior), initial_array_leaves=count,
        production_initial_exactly_matches_fresh_seed_and_probe_initial=True,
        shared_network_training_optimizer_precision_sources_unchanged=True,
        scope='Same192architecture and exact baseline fresh arrays; expanded bank trained cold, no learned baseline/probe import.'
            if capacity_case(case)['control'] == 'bank_growth' else
            'New architecture trained cold, not an import of learned baseline/probe parameters. Same seed policy, different-shaped initial arrays.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('config', 'probe', 'cpu', 'initial'), required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--case', choices=('width192', 'width256_50k', 'width192_bank1024_50k'), default='width192',
                        help='Legacy invocation retains width192; new cases must be selected explicitly')
    args = parser.parse_args()
    if args.action == 'config':
        configuration_audit(args.prior.resolve(), args.output.resolve(), args.case)
    elif args.action == 'probe':
        probe(args.prior.resolve(), args.output.resolve(), args.case)
    elif args.action == 'cpu':
        cpu_replay(args.output.resolve())
    else:
        if args.run is None or args.probe is None:
            parser.error('--run and --probe are required for initial audit')
        initial_audit(args.prior.resolve(), args.run.resolve(), args.probe.resolve(), args.output.resolve(), args.case)
