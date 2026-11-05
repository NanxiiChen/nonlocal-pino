"""Cold, label-free material-conditioned corrosion PINO training.

Run with ``python -m examples.heterogeneous_corrosion2d.train --output RUN``.
The optional --reference NPZ is read exclusively by the reporting evaluator.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import secrets
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.heterogeneous_corrosion2d.evaluate import load_reference, make_evaluator, render_reports
from examples.heterogeneous_corrosion2d.losses import LOSS_NAMES, make_loss_vector, make_step
from examples.heterogeneous_corrosion2d.network import build_network
from examples.heterogeneous_corrosion2d.problem import (
    boundary_mask, grid, initial_fields, material_field, material_image,
    parameter_count, sample_parameters, validate,
)


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def _validate_training(c):
    validate(c)
    if c.train.compute_dtype != 'float32' or c.nl.method not in ('sparse', 'local_ad'):
        raise ValueError('Select FP32 with explicit sparse or local_ad spatial derivatives')
    names = _loss_names(c)
    for label, value in (('iterations', c.train.iterations), ('Nt', c.sample.Nt),
                         ('K_core', c.sample.K_core), ('K_pool', c.sample.K_pool),
                         ('ic_points', c.operator.ic_points), ('condition_batch', c.operator.condition_batch),
                         ('log_every', c.log.log_every), ('eval_every', c.log.eval_every),
                         ('checkpoint_every', c.log.checkpoint_every),
                         ('gradnorm_update_every', c.train.gradnorm_update_every)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{label} must be a positive integer')
    if max(c.sample.K_core, c.operator.ic_points) > c.sample.Nx*c.sample.Ny:
        raise ValueError('Core and initial sample sizes must fit the spatial grid')
    scales = (c.sample.t_max, c.nl.delta, c.train.lr_init, c.train.lr_min,
              c.train.grad_clip, c.loss.ac_scale, c.loss.ch_scale, c.loss.gradnorm_weight_cap)
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError('Time, horizon, optimizer and loss scales must be finite and positive')
    if (len(c.loss.gradnorm_bias) != len(names) or
            not np.isfinite(c.loss.gradnorm_bias).all() or min(c.loss.gradnorm_bias) <= 0):
        raise ValueError(f'Exactly {len(names)} positive loss biases are required')
    if not 0 <= c.train.alpha_gradnorm <= 1:
        raise ValueError('Inverse-gradient EMA new-fraction must be within [0,1]')
    if tuple(c.arch.extent) != (c.geometry.length, c.geometry.height) or c.arch.local_dim != 0:
        raise ValueError('Architecture must use the physical domain and no decoder-local features')


def _loss_names(c):
    if c.nl.method == 'local_ad':
        from examples.heterogeneous_corrosion2d.local_ad import LOSS_NAMES as local_names
        return local_names
    return LOSS_NAMES


def _build_batches(op, points, mouth, ic_ids, c):
    """Each uniformly sampled core has every neighbor; padding has zero weight."""
    batches, core_ids, core_masks, counts = [], [], [], []
    bc_ids = np.flatnonzero(mouth)
    for ids, sub, core, count in op.build_subgraph_pool(c.sample.K_core, c.sample.K_pool):
        ids_host = np.asarray(ids)
        mask = np.asarray(core) & ~mouth[ids_host]
        if not mask.any():
            raise ValueError('A spatial batch contains no free PDE core nodes')
        node_ids = np.r_[ids_host, ic_ids, bc_ids].astype(np.int32)
        batches.append((node_ids, jnp.asarray(points[node_ids], dtype=jnp.float32),
                        tuple(jnp.asarray(a, dtype=jnp.int32 if i < 2 else jnp.float32)
                              for i, a in enumerate(sub)), jnp.asarray(mask)))
        selected = ids_host[mask]
        core_ids.append(np.pad(selected, (0, c.sample.K_core-len(selected)), mode='edge'))
        core_masks.append(np.arange(c.sample.K_core) < len(selected))
        counts.append(count)
    return batches, dict(core_ids=np.stack(core_ids), core_mask=np.stack(core_masks),
                         actual_halo_nodes=np.asarray(counts), ic_ids=ic_ids, mouth_ids=bc_ids)


def _checkpoint(run, step, net, state, weights):
    """Save one model/optimizer pair and publish its completion record last."""
    for name, tree in ((f'model_{step:06d}.eqx', net), (f'optimizer_{step:06d}.eqx', state)):
        target = run/name
        temporary = run/(name+'.partial')
        eqx.tree_serialise_leaves(temporary, tree)
        temporary.replace(target)
    _write_json(run/f'checkpoint_{step:06d}.json', dict(step=step, weights=np.asarray(weights).tolist()))


def train(c, output, reference_path, config_path, paired_run, allow_learning_rate_change):
    started = time.perf_counter()
    _validate_training(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    xx, yy, points = grid(c)
    mouth = boundary_mask(c, points)
    if not mouth.any():
        raise ValueError('The declared grid does not resolve the reservoir mouth')
    # The AD path only uses this graph to reproduce the baseline's core pool;
    # its residual has no graph operation or halo evaluation.
    op = build_D(jnp.asarray(points, dtype=jnp.float32), c.nl.delta,
                 method='sparse', kernel=c.nl.kernel)
    graph = (np.asarray(op._row_idx), np.asarray(op._col_idx), -np.asarray(op._vals))
    reference = None if reference_path is None else load_reference(reference_path, c, points)
    if reference is not None and reference['backend'].item() == 'native_nonlocal_fp32_weights_fp64_integration':
        reference = load_reference(reference_path, c, points, graph)
    # Runtime provenance, not a tuned strategy parameter or configuration field.
    control = None
    if paired_run is not None:
        from examples.heterogeneous_corrosion2d.replay import load_control
        control = load_control(paired_run, c, points, graph,
                               allow_learning_rate_change=allow_learning_rate_change)
    elif allow_learning_rate_change:
        raise ValueError('--allow-learning-rate-change requires --paired-run')
    seed = secrets.randbits(32) if control is None else control[0]
    rng_ic, rng_material, rng_time = [np.random.default_rng(child)
                                    for child in np.random.SeedSequence(seed).spawn(3)]
    ic_ids = rng_ic.choice(len(points), c.operator.ic_points, replace=False).astype(np.int32)
    if control is not None and not np.array_equal(ic_ids, control[1]):
        raise ValueError('Paired IC RNG does not reproduce the saved IC indices')
    initial_ic = jnp.asarray(initial_fields(c, points[ic_ids]), dtype=jnp.float32)
    batches, spatial_trace = _build_batches(op, points, mouth, ic_ids, c)
    if control is not None:
        from examples.heterogeneous_corrosion2d.replay import check_spatial
        check_spatial(spatial_trace, control[4])
    net = build_network(c, jax.random.split(jax.random.PRNGKey(seed), 2)[0])
    if paired_run is not None:
        net = eqx.tree_deserialise_leaves(Path(paired_run)/'model_initial.eqx', net)
    optimizer = make_optimizer(c)
    state = optimizer.init(eqx.filter(net, eqx.is_array))
    if paired_run is not None:
        old = json.loads((Path(paired_run)/'config.json').read_text())
        if old['train']['optimizer'] == c.train.optimizer:
            loaded = eqx.tree_deserialise_leaves(Path(paired_run)/'optimizer_initial.eqx', state)
            for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(state), strict=True):
                if not np.array_equal(np.asarray(a), np.asarray(b)):
                    raise ValueError('Paired optimizer_initial is not the fresh optimizer state')
            state = loaded
    n_graph = len(batches[0][-1])
    boundary_trace = {}
    if c.nl.method == 'local_ad':
        from examples.heterogeneous_corrosion2d.boundaries import neumann_quadrature
        from examples.heterogeneous_corrosion2d.local_ad import make_loss_vector as local_loss
        walls, normals, line_weights = neumann_quadrature(c)
        n_graph = c.sample.K_core
        batches = []
        for ids, mask in zip(spatial_trace['core_ids'], spatial_trace['core_mask'], strict=True):
            node_ids = np.r_[ids, ic_ids, spatial_trace['mouth_ids']]
            queries = np.concatenate((points[node_ids], walls))
            batches.append((node_ids, jnp.asarray(queries, dtype=jnp.float32), None, jnp.asarray(mask)))
        loss = local_loss(c, n_graph, len(ic_ids), int(mouth.sum()), len(walls), normals, line_weights)
        boundary_trace = dict(neumann_points=walls, neumann_normals=normals, neumann_weights=line_weights)
    else:
        loss = make_loss_vector(c, n_graph, len(ic_ids), int(mouth.sum()))
    loss_names = _loss_names(c)
    step = make_step(loss, optimizer, c, loss_names)
    weights = jnp.ones(len(loss_names), dtype=jnp.float32)
    run = Path(output).resolve()
    run.mkdir(parents=True, exist_ok=False)
    _write_json(run/'config.json', json.loads(c.to_json()))
    _write_json(run/'runtime.json', dict(seed=seed, backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()], compilation=c.train.gpu_policy,
        reference_path=None if reference_path is None else str(Path(reference_path).resolve()),
        reference_backend=None if reference is None else reference['backend'].item(),
        spatial_derivative=c.nl.method, optimizer=c.train.optimizer,
        paired_run=None if paired_run is None else str(Path(paired_run).resolve()),
        allow_learning_rate_change=allow_learning_rate_change,
        cold_initialization=True, replay_checks='initial model, fresh optimizer, every input/time and spatial pool',
        sampling='fresh continuous material coefficients and random times every update',
        training_uses_reference_targets=False, checkpoint_selection='fixed training budget'))
    repo = Path(__file__).resolve().parents[2]
    sources = sorted(Path(__file__).parent.rglob('*.py')) + [
        repo/'src/nonlocal_pinn/nn/conditional.py', repo/'src/nonlocal_pinn/operators/__init__.py',
        repo/'src/nonlocal_pinn/training/optimizers.py', repo/'src/nonlocal_pinn/training/runtime.py']
    for source in sources:
        target = run/'source'/source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(config_path, run/'supplied_config.py')
    eqx.tree_serialise_leaves(run/'model_initial.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_initial.eqx', state)
    np.savez_compressed(run/'graph.npz', points=points, graph_row=graph[0],
                        graph_col=graph[1], graph_weight=graph[2], **spatial_trace, **boundary_trace)
    evaluator = None if reference is None else make_evaluator(c, reference)
    history, memories, updates, sample_costs, eval_costs = [], [], [], [], []
    parameter_trace = np.empty((c.train.iterations, c.operator.condition_batch, parameter_count(c)), dtype=np.float32)
    time_trace = np.empty((c.train.iterations, c.sample.Nt), dtype=np.float32)
    prediction, final_metrics = None, None
    training_start = time.perf_counter()

    def evaluate(iteration, model):
        if evaluator is None:
            return None, None
        evaluation_start = time.perf_counter()
        predicted, entry = evaluator(model)
        entry.update(step=iteration, seconds=time.perf_counter()-training_start)
        history.append(entry)
        with (run/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        print(json.dumps({name: value for name, value in entry.items() if name != 'conditions'}, allow_nan=False), flush=True)
        eval_costs.append(time.perf_counter()-evaluation_start)
        return predicted, entry

    prediction, final_metrics = evaluate(0, net)
    with (run/'loss.jsonl').open('w') as loss_stream:
        for iteration in range(c.train.iterations):
            sample_start = time.perf_counter()
            ids, queries, sub, mask = batches[iteration % len(batches)]
            parameters = sample_parameters(c, rng_material, c.operator.condition_batch).astype(np.float32)
            images = material_image(c, parameters, np).astype(np.float32)
            kappa = material_field(c, parameters, points[ids[:n_graph]], np).astype(np.float32)
            times = rng_time.uniform(0, c.sample.t_max, c.sample.Nt).astype(np.float32)
            if control is not None:
                if (not np.array_equal(parameters, control[2][iteration]) or
                        not np.array_equal(times, control[3][iteration])):
                    raise ValueError(f'Paired material/time draw differs at update {iteration+1}')
            parameter_trace[iteration], time_trace[iteration] = parameters, times
            payload = (jnp.asarray(images), jnp.asarray(kappa), queries, initial_ic)
            sample_costs.append(time.perf_counter()-sample_start)
            update_start = time.perf_counter()
            net, state, weights, values, finite = step(net, state, weights, jnp.asarray(times),
                payload, sub, mask, jnp.asarray(iteration % c.train.gradnorm_update_every == 0))
            jax.block_until_ready((net, state, weights, values, finite))
            updates.append(time.perf_counter()-update_start)
            if not bool(finite):
                raise FloatingPointError(f'Nonfinite physics objective/gradient at step {iteration+1}')
            if iteration in (0, 24, c.train.iterations-1):
                stats = jax.devices()[0].memory_stats()
                memories.append(dict(step=iteration+1, allocator=None if stats is None else
                                     {name: int(value) for name, value in stats.items()}))
                _write_json(run/'memory.json', memories)
            if iteration == 0 or (iteration+1) % c.log.log_every == 0 or iteration+1 == c.train.iterations:
                record = dict(step=iteration+1, parts=np.asarray(values).tolist(),
                    loss_names=list(loss_names), weights=np.asarray(weights).tolist(),
                    weighted_loss=float(jnp.sum(values*weights*jnp.asarray(c.loss.gradnorm_bias))),
                    update_seconds=updates[-1], elapsed_seconds=time.perf_counter()-training_start)
                loss_stream.write(json.dumps(record, allow_nan=False)+'\n')
                loss_stream.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
            if (iteration+1) % c.log.checkpoint_every == 0:
                _checkpoint(run, iteration+1, net, state, weights)
            if (iteration+1) % c.log.eval_every == 0 or iteration+1 == c.train.iterations:
                prediction, final_metrics = evaluate(iteration+1, net)
    eqx.tree_serialise_leaves(run/'model_final.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_final.eqx', state)
    np.savez_compressed(run/'sampling.npz', times=time_trace, parameters=parameter_trace)
    if prediction is not None:
        np.savez_compressed(run/'prediction.npz', prediction=prediction, target=reference['fields'],
            parameters=reference['parameters'], time=reference['time'], points=points, X=xx, Y=yy)
        render_reports(run, c, reference, prediction, history)
    _write_json(run/'timing.json', dict(first_update_compile_seconds=updates[0],
        warm_update_mean_seconds=None if len(updates) < 2 else float(np.mean(updates[1:])),
        update_seconds=updates, sample_seconds=sample_costs, evaluation_seconds=eval_costs,
        setup_seconds=training_start-started, total_seconds=time.perf_counter()-started))
    summary = dict(run=str(run), steps=c.train.iterations, seed=seed,
        final=final_metrics, final_weights=np.asarray(weights).tolist(),
        training_uses_reference_targets=False, conditioning='known material image only',
        initial_condition='soft physical phi and c/cle loss',
        boundary=('soft phi=c=0 at mouth; closed walls use explicit phase/q normal losses'
                  if c.nl.method == 'local_ad' else
                  'soft phi=c=0 at mouth; remaining exterior uses missing graph edges'),
        spatial_derivative=c.nl.method, optimizer=c.train.optimizer,
        reference_backend=None if reference is None else reference['backend'].item(),
        paired_run=None if paired_run is None else str(Path(paired_run).resolve()),
        paired_inputs_verified=control is not None, loss_names=list(loss_names),
        residual='original KKS; mobility multiplies full AC driving, constant CH diffusion',
        scope='four held-out material maps; finite-resolution reference, not a continuum certificate',
        total_seconds=time.perf_counter()-started)
    _write_json(run/'summary.json', summary)
    print('COMPLETE '+str(run), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/train_base.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--paired-run', type=Path,
        help='Replay a completed cold run initial model and all input draws; never its final weights')
    parser.add_argument('--allow-learning-rate-change', action='store_true',
        help='Explicit paired LR control: permit only lr_init to differ in addition to declared method changes')
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    spec = importlib.util.spec_from_file_location('heterogeneous_corrosion_config', config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load configuration {config_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.get_config()
    if args.steps is not None:
        c.train.iterations = args.steps
    train(c, args.output, args.reference, config_path, args.paired_run, args.allow_learning_rate_change)


if __name__ == '__main__':
    main()
