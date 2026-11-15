"""Opt-in matched spatial-sampling controls; original NO defaults stay intact."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.icemelting3d.ellipsoids import signed_distance, validate_parameters
from examples.icemelting3d.operator_data import build_network, load_dataset, make_sampler, physics
from examples.icemelting3d.operator_training import build_graph, field_metrics, make_predictor, make_step, write_json
from examples.icemelting3d.spatial_sampling import SpatialSampler, make_accumulator


def monitoring_spec(c, manifest):
    """Explicit opt-in full-test monitoring; preserve historical monitor selection."""
    split = c.operator.monitor_split if 'monitor_split' in c.operator else 'monitor'
    if split not in ('test', 'monitor'):
        raise ValueError('operator.monitor_split must be test or monitor')
    ids = np.asarray(manifest[split+'_indices'])
    if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Monitoring requires nonempty distinct nonnegative integer indices')
    return dict(split=split, indices=ids.tolist(), count=len(ids),
        reference=str(c.operator.dataset_path), evaluation_every=int(c.log.eval_every),
        training_labels=False, checkpoint_selection='Final fixed-budget weights')


def train_spatial(c):
    configure_gpu_policy(c.train.gpu_policy)
    if 'spatial_discretization' in c.operator:
        from examples.icemelting3d.fem_weak import configure_comparison
        configure_comparison(c)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    if c.operator.action not in ('train', 'memory') or jax.default_backend() != 'gpu':
        raise ValueError('Spatial controls require train/memory on GPU; reuse the existing reference bank')
    start = time.perf_counter()
    name = c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-melting-spatial'
    run = Path(c.log.log_dir)/name
    run.mkdir(parents=True, exist_ok=False)
    write_json(run/'config.json', json.loads(c.to_json()))
    if 'spatial_discretization' in c.operator:
        write_json(run/'execution.json', dict(jax=jax.__version__,
            device=jax.devices()[0].device_kind, xla_flags=os.environ['XLA_FLAGS'],
            matmul_precision=jax.config.jax_default_matmul_precision,
            enable_x64=jax.config.jax_enable_x64,
            spatial_discretization=c.operator.spatial_discretization))
    repo = Path(__file__).resolve().parents[2]
    sources = list((repo/'examples/icemelting3d').glob('*.py'))
    sources += list((repo/'examples/icemelting3d/configs').glob('*.py'))
    if 'temporal_query' in c.arch:
        sources += [repo/'examples/electropolish2d/temporal_query.py']
    sources += [repo/p for p in ('src/nonlocal_pinn/nn/conditional.py',
        'src/nonlocal_pinn/training/optimizers.py', 'src/nonlocal_pinn/training/runtime.py',
        'src/nonlocal_pinn/operators/__init__.py')]
    for source in sources:
        target = run/'source'/source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    write_json(run/'source/files.json', [str(p.relative_to(repo)) for p in sources])
    op, points, graph = build_graph(c)
    spatial = SpatialSampler(c, op, np.asarray(points))
    local_ad = ('spatial_discretization' in c.operator
                and c.operator.spatial_discretization == 'local_ad')
    if local_ad:
        graph.update(spatial_discretization='local_ad',
            operator_role='Only preserve canonical core draw/coordinate indexing; no graph in residual',
            boundary='Explicit physical Neumann MSE on all six faces',
            network_pde_queries=c.sample.K_core, initial_jets='analytic unique SDF projection')
    write_json(run/'graph.json', graph)
    keys = jax.random.split(jax.random.key(c.train.seed), 3)
    net = build_network(c, keys[0])
    optimizer = make_optimizer(c); state = optimizer.init(eqx.filter(net, eqx.is_array))
    eqx.tree_serialise_leaves(run/'model_initial.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_initial.eqx', state)
    sampler = make_sampler(c)
    if local_ad:
        from examples.icemelting3d.operator_local_ad import make_initial_jets, make_step as make_ad_step, sample_boundary
        step = make_ad_step(c, optimizer)
        prepare_jets = make_initial_jets(c)
        core_query = eqx.filter_jit(lambda x, m: x[jnp.nonzero(m, size=c.sample.K_core)[0]])
    else:
        step = make_step(c, optimizer)
    accumulate, apply = make_accumulator(c, optimizer)
    width = physics(c)[3]
    initial_fn = eqx.filter_jit(lambda x, p: jax.vmap(
        lambda q: signed_distance(x, q, c.initial.sdf_iterations))(p)/width)
    memory = c.operator.action == 'memory'
    iterations = c.operator.probe_steps if memory else c.train.iterations
    manifest, refs = (None, None) if memory else load_dataset(c)
    monitoring = None if memory else monitoring_spec(c, manifest)
    if monitoring is not None:
        write_json(run/'monitoring.json', monitoring)
    predict = None if memory else make_predictor(c)
    setup = time.perf_counter()-start
    parameters_record = np.empty((iterations, c.operator.condition_batch, 10), dtype=np.float32)
    times_record = np.empty((iterations, c.sample.Nt), dtype=np.float32)
    boundary_record = (np.empty((iterations, c.operator.local_ad.boundary_points, 3), dtype=np.float32)
                       if local_ad else None)
    updates, samples, retrieval, allocations, evaluations = [], [], [], [], []

    def evaluate(indices, iteration, split):
        begin = time.perf_counter(); items = []
        for index in indices:
            ref = refs[index]
            predicted = predict(net, ref['parameters'], ref['points'], ref['time'])
            items.append(dict(index=index, **field_metrics(predicted, ref['phi'])))
            if iteration == iterations:
                np.savez_compressed(run/f'prediction_{index:03d}.npz', prediction=predicted,
                    target=ref['phi'], parameters=ref['parameters'], points=ref['points'], time=ref['time'])
        entry = dict(step=iteration, split=split, conditions=items,
            pooled_phi=float(np.mean([r['pooled_phi'] for r in items])),
            terminal_phi=float(np.mean([r['terminal_phi'] for r in items])),
            seconds=time.perf_counter()-start, evaluation_seconds=time.perf_counter()-begin)
        with (run/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        evaluations.append(entry)
        print('EVALUATION', json.dumps({k: v for k, v in entry.items() if k != 'conditions'}), flush=True)
        return entry

    if refs is not None:
        evaluate(monitoring['indices'], 0, 'monitor')
    loop = time.perf_counter()
    with spatial:
        for k in range(iterations):
            times = jax.random.uniform(jax.random.fold_in(keys[1], k), (c.sample.Nt,),
                minval=c.sample.t_min, maxval=1.)
            condition_key = jax.random.fold_in(keys[2], k)
            sample_seconds = 0.; spatial_seconds = 0.; update_seconds = 0.
            if spatial.mode == 'full':
                grad = jax.tree.map(jnp.zeros_like, eqx.filter(net, eqx.is_array))
                value = jnp.asarray(0., dtype=jnp.float32)
                chunks = iter(spatial.full_chunks())
                for chunk_index in range(len(spatial._host)):
                    begin = time.perf_counter(); (x, sub, mask), weight = next(chunks)
                    spatial_seconds += time.perf_counter()-begin
                    begin = time.perf_counter()
                    if chunk_index == 0:
                        images, u0, parameters = sampler(condition_key, x)
                    else:
                        u0 = initial_fn(x, parameters)
                    jax.block_until_ready((images, u0, parameters))
                    sample_seconds += time.perf_counter()-begin
                    begin = time.perf_counter()
                    grad, value = accumulate(net, grad, value, times, images, u0, x, sub, mask,
                        jnp.asarray(weight, dtype=jnp.float32))
                    jax.block_until_ready((grad, value))
                    update_seconds += time.perf_counter()-begin
                begin = time.perf_counter(); net, state, value, finite = apply(net, state, grad, value)
                jax.block_until_ready((net, state, value, finite)); update_seconds += time.perf_counter()-begin
            else:
                begin = time.perf_counter(); x, sub, mask = spatial.take(k)
                if local_ad:
                    x = core_query(x, mask)
                    jax.block_until_ready(x)
                spatial_seconds = time.perf_counter()-begin
                begin = time.perf_counter(); images, u0, parameters = sampler(condition_key, x)
                if local_ad:
                    boundary, normals = sample_boundary(c, k)
                    boundary_record[k] = boundary
                    boundary, normals = jax.device_put((boundary, normals))
                    jets = prepare_jets(x, boundary, parameters)
                    jax.block_until_ready(jets)
                    if not bool(jnp.all(jets[-1])):
                        raise FloatingPointError(f'Singular/nonfinite initial SDF jets at update{k+1}; no geometry fallback')
                jax.block_until_ready((images, u0, parameters)); sample_seconds = time.perf_counter()-begin
                begin = time.perf_counter()
                if local_ad:
                    net, state, value, finite, parts = step(net, state, times, images, u0, x, jets, boundary, normals)
                else:
                    net, state, value, finite = step(net, state, times, images, u0, x, sub, mask)
                jax.block_until_ready((net, state, value, finite)); update_seconds = time.perf_counter()-begin
            if not bool(finite):
                raise FloatingPointError(f'Nonfinite PDE/gradient at update{k+1}')
            parameters_record[k] = np.asarray(parameters); times_record[k] = np.asarray(times)
            validate_parameters(parameters_record[k])
            updates.append(update_seconds); samples.append(sample_seconds); retrieval.append(spatial_seconds)
            if k == 0 or (k+1) % c.log.log_every == 0 or k+1 == iterations:
                allocator = {name: int(v) for name, v in jax.devices()[0].memory_stats().items()}
                allocations.append(dict(step=k+1, allocator=allocator))
                record = dict(step=k+1, loss=float(value), seconds=time.perf_counter()-start,
                    update_seconds=update_seconds, sample_seconds=sample_seconds, spatial_seconds=spatial_seconds,
                    peak_active_GiB=allocator['peak_bytes_in_use']/1024**3)
                if local_ad:
                    record.update(pde=float(parts[0]), physical_neumann=float(parts[1]))
                with (run/'loss.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                print('TRAIN', json.dumps(record), flush=True)
            if 'model_every' in c.log and (k+1) % c.log.model_every == 0:
                eqx.tree_serialise_leaves(run/f'model_{k+1:06d}.eqx', net)
            if refs is not None and ((k+1) % c.log.eval_every == 0 or k+1 == iterations):
                evaluate(monitoring['indices'], k+1, 'monitor')
    elapsed = time.perf_counter()-loop
    eqx.tree_serialise_leaves(run/'model_final.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_final.eqx', state)
    np.savez_compressed(run/'sampling.npz', parameters=parameters_record, times=times_record, **spatial.trace())
    if local_ad:
        np.savez_compressed(run/'boundary_sampling.npz', points=boundary_record,
                            normals=sample_boundary(c, 0)[1])
    write_json(run/'spatial_sampling.json', spatial.summary())
    test = None if refs is None else evaluate(manifest['test_indices'], iterations, 'test')
    write_json(run/'memory.json', allocations)
    timing = dict(setup_seconds=setup, trainer_seconds=elapsed, update_seconds=updates,
        sample_seconds=samples, spatial_seconds=retrieval, evaluation=evaluations)
    write_json(run/'timing.json', timing)
    summary = dict(run=str(run), mode=spatial.mode, iterations=iterations, cold_start=True,
        condition_batch=c.operator.condition_batch, setup_seconds=setup, trainer_seconds=elapsed,
        test=test, graph=graph, spatial=spatial.summary(), positive_time_training_labels=False,
        peak_active_GiB=max(r['allocator']['peak_bytes_in_use'] for r in allocations)/1024**3,
        scope='Original held-out native-reference diagnostic3Dnodes; graph.json identifies the training discretization; not full-grid or continuum L2')
    write_json(run/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)
    return summary
