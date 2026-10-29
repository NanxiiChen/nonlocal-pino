"""Physics-only neural-operator training through the canonical train.py entry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tensorboardX import SummaryWriter

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.causal import CausalWeightor
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.corrosion2d.residual_experiment import (
    boundary_values, field_metrics, grid, make_step, nondim,
)
from examples.corrosion2d.residuals import graph_action
from examples.electropolish2d.operator_data import (
    condition_split, load_dataset, prepare_dataset, sample_condition_indices,
    training_condition_indices, validate_config,
)


def _build_network(architecture, image_shape, key):
    """Build from explicit architecture metadata and the condition image shape."""
    o = architecture
    return ConditionalFieldOperator(
        image_shape=tuple(image_shape), patch_shape=tuple(o.patch_shape),
        extent=tuple(o.extent), width=o.width, encoder_depth=o.encoder_depth,
        heads=o.heads, mlp_ratio=o.mlp_ratio, decoder_depth=o.decoder_depth,
        spatial_features=o.spatial_features, spatial_scale=o.spatial_scale,
        time_features=o.time_features, time_scale=o.time_scale,
        local_dim=2, out_dim=2, conditioning=o.conditioning,
        norm_eps=o.norm_eps, key=key)


def physical_fields(raw, c):
    """KKS mixture with bounded phi and unrestricted liquid concentration s."""
    phi = .5 * (1 + jnp.tanh(raw[..., 0]))
    h = 3*phi**2 - 2*phi**3
    liquid = c.physical.c_le * raw[..., 1]
    return jnp.stack([phi, liquid + (c.physical.c_se-c.physical.c_le)*h], -1)


def _query_initial(initial, c):
    """Optional feature ablation; prescribed IC targets are never changed.

    Old saved configs have no local_input field and retain initial features.
    """
    if 'local_input' in c.arch and c.arch.local_input == 'zero':
        return jnp.zeros_like(initial)
    return initial


def predict_fields(net, image, points, times, initial, c):
    return physical_fields(net(image, points, times, _query_initial(initial, c)), c)


def make_loss_components(c, nd, n_ic, n_bc):
    """Return five losses plus per-condition causal diagnostics.

    Payload: image, local IC at graph halo/IC/BC queries, and shared coordinates;
    causal mode adds the explicit dynamic two-component eps state.
    Positive-time reference fields are absent from this API.
    """
    n_ch, n1, n2, n3, dc, cle = nd
    causal = CausalWeightor(n_chunks=None) if c.loss.use_causal else None

    def components(net, times, payload, sub_data, core_mask):
        if causal is None:
            images, local, x = payload
        else:
            images, local, x, eps = payload
        n_graph = len(core_mask)

        def condition_parts(image, known):
            prepared = net.prepare(image, x, _query_initial(known, c))
            evolution_cache = jnp.concatenate([prepared[:n_graph], prepared[-2*n_bc:]])

            def physical_at_times(ts):
                return physical_fields(net.decode(evolution_cache, ts), c)

            fields, fields_t = jax.jvp(physical_at_times, (times,), (jnp.ones_like(times),))
            phi, conc = fields[:, :n_graph, 0].T, fields[:, :n_graph, 1].T
            q = conc - (3*phi**2 - 2*phi**3) * dc - cle
            ac = (fields_t[:, :n_graph, 0].T + n3 * graph_action(phi, sub_data)
                  + n2 * (4*phi**3 - 6*phi**2 + 2*phi)
                  - n1 * q * dc * (6*phi - 6*phi**2))
            ch = fields_t[:, :n_graph, 1].T + n_ch * graph_action(q, sub_data)
            mask = core_mask.astype(ac.dtype)[:, None]
            count = jnp.sum(mask) * len(times)
            ac_loss = c.loss.ac_scale * jnp.sum(jnp.abs(ac)*mask) / count
            ch_loss = c.loss.ch_scale * jnp.sum(jnp.abs(ch)*mask) / count
            if causal is None:
                diagnostics = jnp.empty((0, len(times)), dtype=ac.dtype)
            else:
                # Normalize by genuine free-core nodes; padded halo is not data.
                order = jnp.argsort(times)
                ac_segments = c.loss.ac_scale * jnp.sum(jnp.abs(ac)*mask, axis=0)[order] / jnp.sum(mask)
                ch_segments = c.loss.ch_scale * jnp.sum(jnp.abs(ch)*mask, axis=0)[order] / jnp.sum(mask)
                eps_shared = jnp.min(eps)
                w_ac = causal.compute_weights(ac_segments, eps_shared)
                w_ch = causal.compute_weights(ch_segments, eps_shared)
                ac_loss = jnp.mean(w_ac * ac_segments)
                ch_loss = jnp.mean(w_ch * ch_segments)
                diagnostics = jnp.stack([w_ac, w_ch, ac_segments, ch_segments])
            zero = fields[:, n_graph:n_graph+n_bc]
            one = fields[:, n_graph+n_bc:]
            initial = physical_fields(net.decode(prepared[n_graph:n_graph+n_ic],
                                                 jnp.zeros(1, dtype=times.dtype))[0], c)
            parts = jnp.stack([ac_loss, ch_loss,
                jnp.mean(jnp.sum(zero**2, -1)),
                jnp.mean(jnp.sum((one-1)**2, -1)),
                jnp.mean(jnp.sum((initial-known[n_graph:n_graph+n_ic])**2, -1))])
            return parts, diagnostics

        parts, diagnostics = jax.vmap(condition_parts)(images, local)
        return parts.mean(0), diagnostics

    return components


def make_loss_vector(c, nd, n_ic, n_bc):
    """Five-part loss API; noncausal payload/arithmetic remain unchanged."""
    components = make_loss_components(c, nd, n_ic, n_bc)
    return lambda net, times, payload, sub, mask: components(net, times, payload, sub, mask)[0]


def _make_causal_step(components, optimizer, c):
    """Shared causal policy with the existing inverse-L1 balancing rule.

    Aux comes from the same pre-update forward pass as the VJP. The noncausal
    branch continues using corrosion's original make_step unchanged.
    """
    controller = CausalWeightor(n_chunks=None)
    bias = jnp.asarray(c.loss.gradnorm_bias)

    @eqx.filter_jit
    def step(net, state, weights, eps, times, payload, sub, mask, update_weights):
        values, pullback, aux = eqx.filter_vjp(
            lambda n: components(n, times, (*payload, eps), sub, mask), net, has_aux=True)

        def refresh(_):
            norms = []
            for i in range(5):
                grad = pullback(jax.nn.one_hot(i, 5))[0]
                norms.append(sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)))
            norms = jnp.clip(jnp.stack(norms), 1e-6, 1e6)
            proposed = jnp.clip(jnp.mean(norms)/norms, 1e-6, c.loss.gradnorm_weight_cap)
            return jax.lax.stop_gradient(c.train.alpha_gradnorm*proposed + (1-c.train.alpha_gradnorm)*weights)

        new_weights = jax.lax.cond(update_weights, refresh, lambda _: weights, operand=None)
        grad = pullback(new_weights*bias)[0]
        mean_weights = aux[:, :2].mean(0)
        cfg = c.causal
        next_eps = jnp.stack([controller.update_eps(mean_weights[i], eps[i],
            cfg.step_size, cfg.max_eps, cfg.max_last_weight, cfg.min_mean_weight) for i in range(2)])
        finite = (jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(aux))
                  & jnp.all(jnp.isfinite(next_eps)) & jnp.all(next_eps > 0)
                  & jnp.all(jnp.isfinite(new_weights)) & jnp.all(new_weights > 0)
                  & jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))
                        for a in jax.tree.leaves(grad) if eqx.is_array(a)])))
        updates, state = optimizer.update(grad, state, net)
        return eqx.apply_updates(net, updates), state, new_weights, next_eps, values, aux, finite

    return step


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def train_operator(c):
    entry_start = time.perf_counter()
    validate_config(c)
    configure_gpu_policy(c.train.gpu_policy)
    if c.operator.reference_action == 'prepare':
        return prepare_dataset(c)
    if c.arch.model != 'conditional_field':
        raise ValueError('Operator training requires arch.model=conditional_field')
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    data, metadata, references = load_dataset(c)
    online = 'online' in c.operator
    local_ad = c.nl.method == 'local_ad'
    n_losses = len(c.loss.gradnorm_bias)
    selected_inputs = np.empty(0, dtype=np.int32) if online else training_condition_indices(c)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    if not c.log.run_name:
        c = c.copy_and_resolve_references()
        c.log.run_name = stamp + '-operator-' + c.arch.conditioning
    run = Path(c.log.log_dir) / c.log.run_name
    run.mkdir(parents=True, exist_ok=False)
    _write_json(run / 'config.json', json.loads(c.to_json()))
    _write_json(run / 'dataset.json', metadata)
    _write_json(run / 'training_selection.json', dict(
        indices=selected_inputs.tolist(),
        surface_ids=data['surface_ids'][selected_inputs].tolist(),
        condition_batch=c.operator.condition_batch,
        source_train_conditions=c.operator.train_conditions, mode='online_continuous' if online else 'finite_subset'))
    repo = Path(__file__).resolve().parents[2]
    sources = ['examples/electropolish2d/train.py',
        'examples/electropolish2d/operator_training.py', 'examples/electropolish2d/operator_data.py',
        'examples/electropolish2d/operator_local.py',
        'examples/electropolish2d/configs/train_operator.py',
        'examples/electropolish2d/configs/train_operator_single.py',
        'examples/electropolish2d/configs/train_operator_online.py',
        'examples/electropolish2d/configs/train_operator_local.py',
        'examples/electropolish2d/configs/train_operator_causal.py',
        'examples/electropolish2d/online_conditions.py',
        'examples/electropolish2d/configs/train_base.py', 'examples/electropolish2d/configs/train_legacy.py',
        'examples/electropolish2d/losses.py', 'examples/corrosion2d/residual_experiment.py',
        'examples/corrosion2d/configs/residual_pilot.py', 'examples/corrosion2d/residuals.py',
        'src/nonlocal_pinn/nn/conditional.py', 'src/nonlocal_pinn/nn/constraints.py',
        'src/nonlocal_pinn/operators/__init__.py', 'src/nonlocal_pinn/training/runtime.py',
        'src/nonlocal_pinn/training/optimizers.py', 'src/nonlocal_pinn/training/causal.py']
    for name in sources:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
    _write_json(run / 'source/files.json', sources)
    xx, yy, points = grid(c)
    x = jnp.asarray(points, dtype=jnp.float32)
    op = build_D(x, c.nl.delta, method='sparse', kernel=c.nl.kernel)
    for ref in references:
        if (not np.array_equal(op._row_idx, ref['graph_row'])
                or not np.array_equal(op._col_idx, ref['graph_col'])
                or not np.array_equal(-op._vals, ref['graph_weight'])):
            raise ValueError('Training/reference graph mismatch')
    zero, one, _ = boundary_values(c, points)
    free = jnp.asarray(~(zero | one))
    rng = np.random.default_rng(c.train.seed)
    ic_ids = rng.choice(len(points), c.operator.ic_points, replace=False)
    bc_ids = np.r_[np.flatnonzero(zero), np.flatnonzero(one)]
    n_bc = int(zero.sum())
    if n_bc != one.sum():
        raise ValueError('Operator prototype requires equal top/bottom node counts')
    side = (np.isclose(points[:, 0], 0.) | np.isclose(points[:, 0], 1.)) & ~(zero | one)
    side_ids = np.flatnonzero(side)
    n_neumann = len(side_ids)
    if local_ad and n_neumann != 2*(c.sample.Ny-2):
        raise ValueError('Local side-wall nodes must exclude top/bottom Dirichlet corners')
    batches, recorded_cores, recorded_masks, graph_sizes = [], [], [], []
    for ids, sub, core, _ in op.build_subgraph_pool(c.sample.K_core, c.sample.K_pool):
        mask = core & free[ids]
        if not np.any(mask):
            raise ValueError('Empty free-node core mask')
        selected = np.asarray(ids)[np.asarray(mask)]
        padded = np.pad(selected, (0, c.sample.K_core-len(selected)), mode='edge')
        selected_mask = np.arange(c.sample.K_core) < len(selected)
        recorded_cores.append(padded)
        recorded_masks.append(selected_mask)
        graph_sizes.append(len(ids))
        if local_ad:
            ids, sub, mask = jnp.asarray(padded), None, jnp.asarray(selected_mask)
            boundary_ids = np.r_[bc_ids, side_ids]
        else:
            boundary_ids = bc_ids
        node_ids = jnp.concatenate([ids, jnp.asarray(ic_ids), jnp.asarray(boundary_ids)])
        batches.append((node_ids, x[node_ids], sub, mask))
    if online:
        from examples.electropolish2d.online_conditions import make_sampler
        online_sample, online_geometry = make_sampler(c)
        online_nearest = [online_geometry(batch[1]) for batch in batches]
    images = jnp.asarray(data['images'])
    initial = jnp.asarray(data['initial'])
    keys = jr.split(jr.PRNGKey(c.train.seed), 3)
    net = _build_network(c.arch, c.operator.image_shape, keys[0])
    eqx.tree_serialise_leaves(run / 'model_initial.eqx', net)
    optimizer = make_optimizer(c)
    state = optimizer.init(eqx.filter(net, eqx.is_array))
    if c.loss.use_causal:
        step = _make_causal_step(make_loss_components(c, nondim(c), len(ic_ids), n_bc), optimizer, c)
        eps = jnp.full(2, c.causal.eps_init, dtype=jnp.float32)
    else:
        if local_ad:
            from examples.electropolish2d.operator_local import make_loss_vector as local_loss_vector
            loss = local_loss_vector(c, nondim(c), len(ic_ids), n_bc, n_neumann)
        else:
            loss = make_loss_vector(c, nondim(c), len(ic_ids), n_bc)
        step = make_step(loss, optimizer, c)
    weights = jnp.ones(n_losses)
    history = []
    memory_records = []
    writer = SummaryWriter(str(run))
    eval_times = jnp.asarray(references[0]['time'], dtype=jnp.float32)
    predict = eqx.filter_jit(lambda model, image, known: predict_fields(
        model, image, x, eval_times, known, c))
    eval_indices = metadata['eval_indices']
    start = time.perf_counter()
    setup_seconds = start-entry_start
    evaluation_seconds = []
    update_seconds, sample_seconds = [], []
    sampled_times = np.empty((c.train.iterations, c.sample.Nt), dtype=np.float32)

    def evaluate(iteration):
        evaluation_start = time.perf_counter()
        records, predictions = [], []
        for i, ref in zip(eval_indices, references):
            pred = np.asarray(predict(net, images[i], initial[i])).reshape(
                len(eval_times), c.sample.Ny, c.sample.Nx, 2)
            if not np.isfinite(pred).all():
                raise FloatingPointError('Nonfinite operator evaluation')
            m = field_metrics(pred, ref)
            phase=pred[...,0]; conc=pred[...,1]
            q=conc-(3*phase**2-2*phase**3)*(c.physical.c_se-c.physical.c_le)-c.physical.c_le
            m.update(condition_index=i, split=('monitor' if i < c.operator.train_conditions else 'test') if online else
                           condition_split(i, c.operator.train_conditions, selected_inputs),
                     phi_min=float(phase.min()), phi_max=float(phase.max()),
                     c_min=float(conc.min()), c_max=float(conc.max()),
                     q_min=float(q.min()), q_max=float(q.max()))
            records.append(m)
            predictions.append(pred)
        entry = dict(step=iteration, seconds=time.perf_counter()-start, conditions=records)
        for split in ('train', 'unused_train', 'monitor', 'test'):
            selected = [m for m in records if m['split'] == split]
            if not selected:
                continue
            for field in ('phi', 'c'):
                for prefix in ('pooled_rel_l2_', 'terminal_rel_l2_'):
                    metric = prefix + field
                    # field_metrics uses pooled_rel_l2_phi/c for the pooled-time metric.
                    entry[split+'_'+metric] = float(np.mean([m[metric] for m in selected]))
        history.append(entry)
        with (run / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        for name, value in entry.items():
            if isinstance(value, (float, int)):
                writer.add_scalar('eval/'+name, value, iteration)
        print(json.dumps({k: v for k, v in entry.items() if k != 'conditions'}), flush=True)
        evaluation_seconds.append(time.perf_counter()-evaluation_start)
        return np.stack(predictions), entry

    prediction, _ = evaluate(0)
    if online:
        online_coefficients = np.empty((c.train.iterations, c.operator.condition_batch,
                                       c.surface.n_modes, 3), dtype=np.float32)
        online_accepted = np.empty(c.train.iterations, dtype=np.int32)
    else:
        sampled_conditions = np.empty((c.train.iterations, c.operator.condition_batch), dtype=np.int32)
    # Reference arrays never enter step/payload. Only prescribed IC values do.
    with (run / 'loss.jsonl').open('w') as loss_stream:
        for iteration in range(c.train.iterations):
            k_data = jr.fold_in(keys[1], iteration)
            k_cond, k_time = jr.split(k_data)
            times = jr.uniform(k_time, (c.sample.Nt,), minval=c.sample.t_min, maxval=1.)
            sampled_times[iteration] = np.asarray(times)
            pool_index = iteration % len(batches)
            node_ids, queries, sub, mask = batches[pool_index]
            sample_start = time.perf_counter()
            if online:
                fresh_images, local, coefficients, accepted = online_sample(
                    jr.fold_in(keys[2], iteration), queries, online_nearest[pool_index])
                if int(accepted) < c.operator.condition_batch:
                    raise ValueError(f'Online accepted only {int(accepted)} valid candidates; '
                                     'increase the explicit candidate_factor; no update was applied')
                payload = (fresh_images, local, queries)
            else:
                cond_ids = sample_condition_indices(k_cond, selected_inputs, c.operator.condition_batch)
                payload = (images[cond_ids], initial[cond_ids][:, node_ids], queries)
            refresh = jnp.asarray(iteration % c.train.gradnorm_update_every == 0)
            sample_seconds.append(time.perf_counter()-sample_start)
            update_start = time.perf_counter()
            if c.loss.use_causal:
                eps_used = eps
                net, state, weights, eps, values, causal_aux, finite = step(
                    net, state, weights, eps, times, payload, sub, mask, refresh)
            else:
                net, state, weights, values, finite = step(net, state, weights, times,
                    payload, sub, mask, refresh, jnp.ones(n_losses, dtype=bool))
            jax.block_until_ready((net, state, weights, values, finite))
            update_seconds.append(time.perf_counter()-update_start)
            if not bool(finite):
                raise FloatingPointError(f'Nonfinite operator gradient/objective at update {iteration+1}')
            if iteration in (0, 24, c.train.iterations-1):
                stats = jax.devices()[0].memory_stats()
                memory_records.append(dict(step=iteration+1, backend=jax.default_backend(),
                    allocator_stats=None if stats is None else {k:int(v) for k,v in stats.items()}))
            if online:
                online_coefficients[iteration] = np.asarray(coefficients)
                online_accepted[iteration] = int(accepted)
            else:
                sampled_conditions[iteration] = np.asarray(cond_ids)
            if (iteration+1) % c.log.log_every == 0 or iteration == 0:
                entry = dict(step=iteration+1, parts=np.asarray(values).tolist(),
                             weights=np.asarray(weights).tolist(), seconds=time.perf_counter()-start)
                if c.loss.use_causal:
                    aux = np.asarray(causal_aux)
                    entry['unweighted_parts'] = np.r_[aux[:, 2:].mean(axis=(0, 2)), np.asarray(values)[2:]].tolist()
                    entry['causal'] = dict(eps_used=np.asarray(eps_used).tolist(), eps_next=np.asarray(eps).tolist(),
                        sorted_times=np.sort(np.asarray(times)).tolist(),
                        weights_ac=aux[:, 0].tolist(), weights_ch=aux[:, 1].tolist(),
                        segments_ac=aux[:, 2].tolist(), segments_ch=aux[:, 3].tolist())
                loss_stream.write(json.dumps(entry, allow_nan=False)+'\n'); loss_stream.flush()
                print(json.dumps(entry), flush=True)
            if (iteration+1) % c.log.eval_every == 0 or iteration+1 == c.train.iterations:
                prediction, last = evaluate(iteration+1)
    elapsed = time.perf_counter()-start
    eqx.tree_serialise_leaves(run / 'model_final.eqx', net)
    np.savez_compressed(run / 'sampling.npz', times=sampled_times,
                        core_ids=np.stack(recorded_cores), core_mask=np.stack(recorded_masks),
                        ic_ids=ic_ids, dirichlet_ids=bc_ids, side_ids=side_ids,
                        nonlocal_halo_counts=np.asarray(graph_sizes))
    warm = np.asarray(update_seconds)[1:]
    refresh_indices = np.arange(1, c.train.iterations) % c.train.gradnorm_update_every == 0
    timing = dict(setup_seconds=setup_seconds, trainer_seconds=elapsed,
        first_update_compile_and_call_seconds=update_seconds[0],
        update_seconds=update_seconds, sample_seconds=sample_seconds,
        evaluation_seconds=evaluation_seconds,
        warm_update_mean_seconds=None if not len(warm) else float(warm.mean()),
        warm_nonrefresh_median_seconds=None if not np.any(~refresh_indices) else float(np.median(warm[~refresh_indices])),
        warm_refresh_median_seconds=None if not np.any(refresh_indices) else float(np.median(warm[refresh_indices])),
        scope='Updates synchronized through parameters/optimizer state; first update includes compilation; setup/evaluation separate')
    _write_json(run / 'timing.json', timing)
    if online:
        np.save(run / 'online_coefficients.npy', online_coefficients)
        np.save(run / 'online_accepted.npy', online_accepted)
    else:
        np.save(run / 'sampled_conditions.npy', sampled_conditions)
    np.savez_compressed(run / 'prediction.npz', X=xx, Y=yy, time=references[0]['time'],
                        condition_indices=eval_indices, surface_ids=data['surface_ids'][eval_indices],
                        prediction=prediction,
                        target=np.stack([np.stack([r['phi'], r['c']], -1) for r in references]))
    summary = dict(run=str(run), seconds=elapsed, final=last, conditioning=c.arch.conditioning,
        iterations=c.train.iterations, dataset_path=c.operator.dataset_path,
        spatial_operator=c.nl.method, loss_count=n_losses,
        boundary=('explicit positive-time phi_n=q_n/c_le=0 on side walls; six losses'
                  if local_ad else 'no external graph edges; five losses'),
        training_uses_positive_time_labels=False, checkpoint_selection='final fixed budget; no test selection',
        training_indices=selected_inputs.tolist(),
        sampling_mode='online_continuous' if online else 'finite_subset',
        sampled_training_indices=[] if online else np.unique(sampled_conditions).tolist(),
        scope=('continuous online initial-interface family; fixed held-out GRAPH diagnostics, not local convergence'
               if local_ad else 'continuous online initial-interface family; fixed held-out reference bank; same-graph error'
               if online else 'single fixed-interface fitting; held-out diagnostics are not operator-generalization evidence'
               if len(selected_inputs) == 1 else
               'small held-out initial-interface family; same-graph error, not continuum convergence'))
    if c.loss.use_causal:
        summary['causal'] = dict(mode='per_condition_per_time', eps_final=np.asarray(eps).tolist(),
            policy='shared min eps, separate AC/CH weights, batch-mean weight adaptation; stopped weights')
    _write_json(run / 'summary.json', summary)
    _write_json(run / 'memory.json', memory_records)
    writer.close()
    print('COMPLETE '+str(run), flush=True)
    return summary
