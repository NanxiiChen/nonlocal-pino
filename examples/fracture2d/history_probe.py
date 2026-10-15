"""Chronological frozen-history prototype; no coupled loading state is changed."""
import json
from pathlib import Path
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.at2_cg import damage_solver
from examples.fracture2d.history_network import (
    HistoryOperator, HistoryReplay, build_patches, make_context)
from examples.fracture2d.losses import damage_residual, residual_mean
from examples.fracture2d.mesh import load_geometry, read_cloud
from examples.fracture2d.optimization import make_optimizer, optimizer_count
from examples.fracture2d.sampling import pack_rows
from examples.fracture2d.storage import atomic_json


def make_loss(*, coarse, fine, points, columns, values, gc0, gc1, floor):
    def image(net, context):
        return net(context, points, coarse, fine, gc0=gc0)

    def single(net, H, ids):
        d = image(net, dict(H=H, fine_ids=ids))
        diffusion = jnp.sum(values*d[columns], axis=1)
        residual = damage_residual(d, H, diffusion, gc0=gc0, gc1=gc1)
        n = H.shape[0]
        return residual_mean(residual, H, jnp.full(n, 1/n, jnp.float64),
            jnp.ones(n, jnp.float64), population_size=n, norm='l1', floor=floor)

    def loss(net, batch):
        batch = jax.tree.map(jax.lax.stop_gradient, batch)
        def body(total, task):
            H, ids, weight = task
            # Skip inactive tasks at runtime while compiling both static branches
            # once. Inactive H is also finite/positive, never a 0/0 placeholder.
            value = jax.lax.cond(weight > 0,
                lambda _: single(net, H, ids), lambda _: jnp.zeros((), jnp.float64), None)
            return total+weight*value, None
        return jax.lax.scan(body, jnp.zeros((), jnp.float64),
            (batch['H'], batch['fine_ids'], batch['weights']))[0]
    return image, loss


def make_step(objective, optimizer, *, old_count, fraction):
    @eqx.filter_jit
    def step(net, state, current, replay, key, enabled):
        batch = replay.batch(current, key, enabled, old_count=old_count, fraction=fraction)
        value, gradient = eqx.filter_value_and_grad(objective)(net, batch)
        updates, state = optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), state, value
    return step


def load_snapshots(c, mechanics):
    """Read two explicitly labeled H inputs per actual load, never all inner states."""
    h = c.history
    root = Path(h.source_root)
    saved = json.loads((root/'config.json').read_text())
    cloud = read_cloud(saved['data']['cloud'])
    n = mechanics.n_material
    for actual, expected in [(cloud['coordinates'], mechanics.coordinates[:n]),
            (cloud['triangles'], mechanics.triangles[:mechanics.n_material_cells]),
            (cloud['volumes'], mechanics.volumes)]:
        if not np.array_equal(actual, expected):
            raise ValueError('history source particle order/geometry differs from configured operator')
    for key in ('Gc', 'l0', 'E', 'nu', 'k_res'):
        if saved['physical'][key] != c.physical[key]:
            raise ValueError(f'history source physical parameter differs: {key}')
    for key in ('delta_d', 'visibility_tolerance'):
        if saved['nl'][key] != c.nl[key]:
            raise ValueError(f'history source operator setting differs: {key}')
    rows = json.loads((root/'result.json').read_text())['curve']
    loads = list(h.loads)
    if (not loads or not np.isfinite(loads+[h.held_out_load]).all()
            or min(loads) <= 0 or np.any(np.diff(loads) <= 0)
            or h.held_out_load <= loads[-1]):
        raise ValueError('strictly increasing positive training loads and later held-out load required')
    snapshots = []
    for load in loads+[h.held_out_load]:
        matches = [row for row in rows if abs(row['load']-load) <= 1e-12]
        if len(matches) != 1 or matches[0]['q_d_synchronized'] is not True:
            raise ValueError(f'one synchronized source endpoint required at actual load {load}')
        row = matches[0]
        folder = root/f"state_{row['index']:04d}"
        for phase, path in [('start', folder/'outer_0001/fields.npz'),
                            ('end', folder/'fields.npz')]:
            with np.load(path, allow_pickle=False) as data:
                H = data['H'].copy()
            if (H.shape != (n,) or H.dtype != np.float64 or not np.isfinite(H).all()
                    or H.min() < 0 or H.max() <= 0):
                raise ValueError(f'invalid positive FP64 snapshot history: {path}')
            snapshots.append(dict(load=float(load), phase=phase, path=str(path), H=H))
    return snapshots[:-2], snapshots[-2:]


def run(c):
    h = c.history
    root = Path(c.output.root)
    if root.exists():
        raise FileExistsError(f'fresh output directory required: {root}')
    if (c.runtime.platform not in ('cpu', 'gpu') or c.loss.norm != 'l1'
            or c.sample.mode != 'full' or c.train.optimizer != 'adam'
            or c.arch.output_head != 'sigmoid' or c.loss.weight_mode != 'uniform'
            or type(h.steps_per_snapshot) is not int or h.steps_per_snapshot < 1
            or type(h.replay_capacity) is not int or h.replay_capacity < 2*len(h.loads)
            or type(h.replay_batch) is not int or h.replay_batch < 1
            or not 0 < h.replay_fraction < 1):
        raise ValueError('history probe requires full/L1/uniform/Adam/sigmoid and explicit fixed replay controls')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platform_name', c.runtime.platform)
    m, operator, _ = load_geometry(c)
    training, held_out = load_snapshots(c, m)
    coarse = build_patches(m.coordinates[:m.n_material], m.volumes, cells=h.coarse_cells)
    fine = build_patches(m.coordinates[:m.n_material], m.volumes, cells=h.fine_cells)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    all_cases = training+held_out
    contexts = [make_context(s['H'], fine, gc0=gc0, count=h.fine_tokens) for s in all_cases]
    low, span = m.coordinates[:m.n_material].min(0), np.ptp(m.coordinates[:m.n_material], axis=0)
    points = jnp.asarray(2*(m.coordinates[:m.n_material]-low)/span-1)
    D = operator.to_scipy()
    columns, values = pack_rows(D, width=int(np.max(np.diff(D.indptr))))
    image, objective = make_loss(coarse=coarse, fine=fine, points=points,
        columns=jnp.asarray(columns), values=jnp.asarray(values), gc0=gc0, gc1=gc1, floor=c.loss.floor)
    image, evaluate_loss = eqx.filter_jit(image), eqx.filter_jit(objective)
    net0 = HistoryOperator(width=h.width, point_depth=h.point_depth,
        query_depth=h.query_depth, head_depth=h.head_depth, heads=h.heads,
        initial_damage=c.arch.initial_damage, key=jax.random.PRNGKey(c.train.seed))
    keys = ('learning_rate', 'decay_steps', 'decay_rate', 'minimum_rate',
        'grad_clip', 'b1', 'b2', 'eps', 'eps_root', 'shampoo_beta',
        'correct_bias', 'precondition_frequency', 'max_precond_dim',
        'precondition_1d', 'weight_decay')
    optimizer, _ = make_optimizer(kind=c.train.optimizer, **{k: c.train[k] for k in keys})
    step = make_step(objective, optimizer, old_count=h.replay_batch, fraction=h.replay_fraction)
    replay0 = HistoryReplay(capacity=h.replay_capacity, nodes=m.n_material, fine_count=h.fine_tokens)
    root.mkdir(parents=True)
    atomic_json(root/'config.json', c.to_dict())
    import shutil
    snapshot_dir = root/'code_snapshot'
    snapshot_dir.mkdir()
    for path in (Path(__file__), Path(__file__).with_name('history_network.py'),
                 Path(__file__).with_name('configs')/'history_probe.py'):
        name = 'history_probe_config.py' if path.parent.name == 'configs' else path.name
        shutil.copyfile(path, snapshot_dir/name)
    source = [dict(load=s['load'], phase=s['phase'], path=s['path']) for s in all_cases]
    report = dict(complete=False, source=source, training_snapshots=len(training),
        material_nodes=m.n_material, backend=jax.default_backend(), device=str(jax.devices()[0]),
        parameters=sum(a.size for a in jax.tree.leaves(eqx.filter(net0, eqx.is_inexact_array))),
        shapes=dict(bank=list(replay0.histories.shape), task_batch=[1+h.replay_batch, m.n_material],
            coarse_groups=list(coarse.indices.shape), fine_bank=list(fine.indices.shape),
            tokens=h.coarse_cells**2+h.fine_tokens),
        replay_collection='first_outer_H_then_synchronized_endpoint_H_only',
        direct_used_for_training=False, arms={})
    atomic_json(root/'result.json', report)
    oracle = damage_solver(operator, m.volumes, gc0, gc1, tol=c.validation.cg_rtol,
        atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)
    oracle_fields = [np.asarray(oracle(jnp.asarray(s['H']), jnp.zeros(m.n_material))) for s in all_cases]
    def metrics(net, index):
        pred = np.asarray(image(net, contexts[index]))
        ctx = contexts[index]
        batch = dict(H=ctx['H'][None], fine_ids=ctx['fine_ids'][None], weights=jnp.ones(1))
        loss = float(evaluate_loss(net, batch))
        relative = np.sqrt(np.sum(m.volumes*(pred-oracle_fields[index])**2)
                           /np.sum(m.volumes*oracle_fields[index]**2))
        if not np.isfinite([loss, relative]).all():
            raise FloatingPointError('nonfinite history probe evaluation')
        return dict(loss=loss, damage_relative_l2=float(relative))
    started = time.perf_counter()
    state0 = optimizer.init(eqx.filter(net0, eqx.is_inexact_array))
    step(net0, state0, contexts[0], replay0, jax.random.PRNGKey(123), jnp.asarray(False))[2].block_until_ready()
    report['initial_compile_seconds'] = time.perf_counter()-started
    # Both controls use the exact same compiled update and independent cold Adam.
    for enabled in (False, True):
        name = 'residual_replay' if enabled else 'current_only'
        net, state, replay = net0, state0, replay0
        records = []
        for index, snapshot in enumerate(training):
            before = metrics(net, index)
            started = time.perf_counter()
            active_replay = enabled and int(replay.count) > 0
            for update in range(h.steps_per_snapshot):
                key = jax.random.fold_in(jax.random.PRNGKey(c.train.seed+1729),
                                         index*h.steps_per_snapshot+update)
                net, state, loss = step(net, state, contexts[index], replay, key, jnp.asarray(enabled))
            loss.block_until_ready()
            seconds = time.perf_counter()-started
            if not np.isfinite(float(loss)):
                raise FloatingPointError('nonfinite cross-history training objective')
            expected = (index+1)*h.steps_per_snapshot
            if optimizer_count(state, kind='adam') != expected:
                raise RuntimeError('Adam count reset across histories')
            record = dict(load=snapshot['load'], phase=snapshot['phase'], before=before,
                after=metrics(net, index), updates=h.steps_per_snapshot,
                active_field_evaluations=h.steps_per_snapshot*(1+h.replay_batch if active_replay else 1),
                training_seconds=seconds, prior_snapshots=int(replay.count))
            records.append(record)
            # Exactly one insertion per load phase; no insertion inside Adam.
            replay = replay.append(contexts[index])
            report['arms'][name] = dict(records=records, complete=False)
            atomic_json(root/'result.json', report)
            print(json.dumps(dict(arm=name, **record)), flush=True)
        folder = root/name
        folder.mkdir()
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        eqx.tree_serialise_leaves(folder/'optimizer.eqx', state)
        eqx.tree_serialise_leaves(folder/'replay.eqx', replay)
        report['arms'][name] = dict(complete=True, records=records,
            total_training_seconds=sum(r['training_seconds'] for r in records),
            retained=[metrics(net, i) for i in range(len(training))],
            held_out_before_any_fit=[dict(load=s['load'], phase=s['phase'], **metrics(net, len(training)+i))
                                     for i, s in enumerate(held_out)])
        atomic_json(root/'result.json', report)
    report['complete'] = True
    report['interpretation'] = ('Bounded same-architecture equal-update pilot, with unequal replay work; '
        'no matched-accuracy speedup, MLP superiority or coupled-fracture claim.')
    atomic_json(root/'result.json', report)
    return report


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'history_probe.py'),
                                   'Optional cross-history point-encoder probe', lock_config=True)
    def _main(_):
        run(flags.FLAGS.config)
    app.run(_main)


if __name__ == '__main__':
    main()
