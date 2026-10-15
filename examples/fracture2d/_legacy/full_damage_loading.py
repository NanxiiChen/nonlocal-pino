"""Cold all-node L1/Adam mechanics-first loading with explicit load-step controls."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.core_halo_damage import pack_rows
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config
from examples.fracture2d._legacy.l1_damage_loading import committed_history, residual_metrics
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import make_loss_fn, make_step, optimizer_count
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.plain_sigmoid_damage import latent_basis_scaling
from examples.fracture2d._legacy.provisional_damage import mesh_adjacency, crack_extent
from examples.fracture2d._legacy.staggered_damage import mechanics_first_sweep, field_changes, field_stop


def load_grid(a):
    """Allow equal pre/post increments; construct endpoints by integer counts."""
    values = [a.pre_step, a.post_step, a.transition_load, a.end_load]
    if (not np.isfinite(values).all() or min(values) <= 0
            or a.end_load <= a.transition_load or a.pre_step < a.post_step):
        raise ValueError('invalid explicit coarse/fine cold load grid')
    n = round(a.transition_load/a.pre_step)
    k = round((a.end_load-a.transition_load)/a.post_step)
    if (n < 1 or k < 1
            or not np.isclose(n*a.pre_step, a.transition_load, rtol=0, atol=1e-14)
            or not np.isclose(a.transition_load+k*a.post_step, a.end_load, rtol=0, atol=1e-14)):
        raise ValueError('requested loads must lie on integer increment grids')
    return np.concatenate((np.arange(1, n+1)*a.pre_step,
        a.transition_load+np.arange(1, k+1)*a.post_step)).astype(np.float64)


def common_config(c):
    """Only case identity and precritical load step differ in the paired experiment."""
    settings = c.to_dict()
    del settings['full_loading']['case'], settings['full_loading']['pre_step']
    return settings


def validate(c):
    a, t = c.full_loading, c.l1_loading
    grid = load_grid(a)
    if (a.case not in ('coarse', 'fine') or a.sampling != 'all_nodes'
            or a.symmetry_constraint is not False or c.stagger.omega != 1.
            or a.advance_on_budget_exit is not True or a.budget_exit_is_certified is not False
            or a.stop_on_field_stability is not True or t.loss_norm != 'l1'
            or t.optimizer != 'adam' or t.output_head != 'sigmoid'
            or t.history_feature_scaling != 'physical' or t.parameterization != 'latent_basis_rms'
            or t.trunk_gain != 1. or t.initial_physical_damage != 0.
            or t.initial_physical_history != 0. or t.decay_rate != 1.
            or t.minimum_rate != t.learning_rate or not 0 < c.head.initial_damage < 1.
            or a.critical_load != a.end_load
            or min(a.damage_change_limit, a.displacement_change_limit) <= 0
            or not np.isfinite([a.damage_change_limit, a.displacement_change_limit]).all()
            or any(type(v) is not int or v < 1 for v in
                (a.inner_updates, a.prefix_outer_cap, a.critical_outer_cap,
                 a.minimum_sweeps, a.consecutive_passes, a.cpu_threads))
            or a.minimum_sweeps > a.prefix_outer_cap
            or any(type(v) is not int or not 1 <= v <= a.critical_outer_cap
                   for v in a.checkpoint_sweeps)):
        raise ValueError('invalid explicit cold all-node stagger protocol')
    return grid


def run(c, config_path):
    grid = validate(c)
    a, t = c.full_loading, c.l1_loading
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise ValueError('cold all-node loading requires serialized FP64 GPU')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    suite = Path(a.suite_root); root = suite/a.case
    if (root/'identity.json').exists():
        raise ValueError('refusing to overwrite existing cold loading branch')
    root.mkdir(parents=True, exist_ok=True)
    m, op, geometry = load_geometry(c)
    parity = json.loads((Path(c.trial.geometry_dir)/'fenics_parity.json').read_text())
    if not parity['pass'] or parity['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('prepared local FEM geometry has not passed parity validation')
    if sha(t.basis_path) != t.basis_sha256:
        raise ValueError('geometry-only compact basis hash differs')
    B, D = load_npz(t.basis_path), op.to_scipy()
    if B.shape[0] != m.n_material:
        raise ValueError('basis and material-node ordering differ')
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    features, info = make_features(op, m.coordinates[:m.n_material], gc0=g0,
        length=c.physical.l0, tip=list(c.conditioned.tip), powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    features = eqx.filter_jit(features)
    net = fresh_network(c, info['input_dimension'], B, head=t.output_head)
    net, scales = latent_basis_scaling(net, B, np.ones(B.shape[1]))
    optimizer, schedule = adam_from_config(t)
    state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
    bc, bv = pack_rows(B@diags(scales), width=int(np.max(np.diff(B.indptr))))
    dc, dv = pack_rows(D, width=int(np.max(np.diff(D.indptr))))
    bc, bv, dc, dv = map(jnp.asarray, (bc, bv, dc, dv))
    local = jnp.arange(m.n_material, dtype=jnp.int32)
    probability = jnp.full(m.n_material, 1/m.n_material, jnp.float64)
    def context(H):
        H = jax.lax.stop_gradient(H)
        return dict(H=H, core_H=H, features=jax.lax.stop_gradient(features(H)),
            basis_columns=bc, basis_values=bv, core_local=local,
            operator_columns=dc, operator_values=dv, p_core=probability)
    image, objective = make_loss_fn(gc0=g0, gc1=g1, gain=t.trunk_gain,
        population_size=m.n_material, norm=t.loss_norm)
    image, value = eqx.filter_jit(image), eqx.filter_jit(objective)
    step = make_step(objective, optimizer)
    solve = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
        maxiter=c.validation.cg_maxiter)
    eqx.tree_serialise_leaves(root/'initial_network.eqx', net)
    eqx.tree_serialise_leaves(root/'initial_optimizer.eqx', state)
    atomic_fields(root/'parameterization.npz', column_scale=scales, trunk_gain=np.float64(t.trunk_gain))
    init = dict(common_config=common_config(c), network_sha256=sha(root/'initial_network.eqx'),
        optimizer_sha256=sha(root/'initial_optimizer.eqx'))
    common = suite/'common_initialization.json'
    if a.case == 'coarse':
        if common.exists(): raise ValueError('paired common initialization already exists')
        atomic_json(common, init)
    elif not common.exists() or json.loads(common.read_text()) != init:
        raise ValueError('fine branch differs from the common cold configuration/initial weights')
    repo = Path(__file__).resolve().parents[3]
    protected = {str(Path(t.basis_path)): sha(t.basis_path)}
    for name in ('identity.json', 'mesh.npz', 'damage_operator.npz', 'fenics_parity.json'):
        path = Path(c.trial.geometry_dir)/name; protected[str(path)] = sha(path)
    code = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):
                code[str(path.relative_to(repo))] = sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))] = sha(config_path)
    for name in code:
        path = root/'code_snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, path)
    atomic_json(root/'identity.json', dict(schema='cold-all-node-l1-adam-v1', config=c.to_dict(),
        loads=grid.tolist(), geometry=geometry, features=info, code=code, source=protected,
        common_initialization=init, sampling='all_nodes', full_node_count=m.n_material,
        own_zero=True, reference_training_targets=False, symmetry_constraint=False,
        optimizer_persistent=True, history_floor='preceding_load_endpoint',
        residual_is_stopping_gate=False, provisional_load_advancement=True))
    q, d = jnp.zeros(len(m.load_direction), jnp.float64), jnp.zeros(m.n_material, jnp.float64)
    H = jnp.zeros_like(d)
    zero = root/'state_0000'; zero.mkdir()
    atomic_fields(zero/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H))
    atomic_json(zero/'result.json', dict(load=0., reaction=0., status='exact_physical_zero',
        optimizer_count=0, certified_path=True, fields_sha256=sha(zero/'fields.npz')))
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    adjacency = mesh_adjacency(tri, population_size=m.n_material)
    count = 0; curve = []; started = time.perf_counter()

    def train(history, old_damage, carry):
        del old_damage
        net, memory = carry; ctx = context(history); tic = time.perf_counter()
        for _ in range(a.inner_updates):
            net, memory, loss, grad = step(net, memory, ctx, jnp.asarray(load, jnp.float64))
        output = image(net, ctx); output.block_until_ready()
        m.factors(np.asarray(output))
        if not np.isfinite([float(loss), float(grad)]).all():
            raise FloatingPointError('nonfinite all-node Adam objective/gradient')
        raw, relative = residual_metrics(np.asarray(output), np.asarray(history), D, gc0=g0, gc1=g1)
        mae = float(abs(raw).mean()/(2*np.sqrt(np.mean(np.asarray(history)**2))))
        np.testing.assert_allclose(value(net, ctx, jnp.asarray(load)), mae, rtol=1e-10, atol=1e-12)
        return output, (net, memory), dict(full_loss=mae, AT2_relative=relative,
            iterations=a.inner_updates, seconds=time.perf_counter()-tic)

    def checkpoint(folder, fields, row, carry):
        folder.mkdir(exist_ok=True)
        atomic_fields(folder/'fields.npz', **fields)
        eqx.tree_serialise_leaves(folder/'network.eqx', carry[0])
        eqx.tree_serialise_leaves(folder/'optimizer.eqx', carry[1])
        row.update(fields_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
            optimizer_sha256=sha(folder/'optimizer.eqx'))
        atomic_json(folder/'report.json', row)

    carry = (net, state)
    for index, target in enumerate(grid, 1):
        load = float(target); parent_d, parent_H = d, H
        folder = root/f'state_{index:04d}'; folder.mkdir()
        parent = root/f'state_{index-1:04d}'/'fields.npz'
        parent_count = count; consecutive = 0; load_started = time.perf_counter()
        cap = a.critical_outer_cap if abs(load-a.critical_load) < 1e-14 else a.prefix_outer_cap
        for sweep in range(1, cap+1):
            tic = time.perf_counter()
            fields, carry = mechanics_first_sweep(q, d, parent_H, load, carry,
                mechanics=solve, driver=m.driver, damage=train, omega=1.)
            q, d, H = (fields[k] for k in ('q', 'd', 'H'))
            count += a.inner_updates
            if optimizer_count(carry[1], kind='adam') != count:
                raise RuntimeError('Adam memory reset across outer iterations or loads')
            changes = field_changes(q, fields['q_previous'], d, fields['d_previous'], tri, area, load=load)
            stop = field_stop(changes, displacement_limit=a.displacement_change_limit,
                damage_limit=a.damage_change_limit, sweep=sweep, minimum_sweeps=a.minimum_sweeps,
                consecutive_passes=a.consecutive_passes, previous_consecutive=consecutive)
            consecutive = stop['consecutive']
            row = dict(index=index, load=load, sweep=sweep, outer_cap=cap, **changes,
                field_stopping=stop, field_stable=stop['converged'], reaction=fields['reaction'],
                q_d_synchronized=False, mechanics_calls=sweep, mechanics=fields['mechanics'],
                damage_min=float(d.min()), damage_max=float(d.max()), inner_AT2=fields['inner']['AT2_relative'],
                full_loss=fields['inner']['full_loss'], training_updates=sweep*a.inner_updates,
                optimizer_count=count, parent_optimizer_count=parent_count,
                learning_rate=float(schedule(count-1)), parent_sha256=sha(parent),
                seconds=time.perf_counter()-load_started, sweep_seconds=time.perf_counter()-tic,
                training_seconds=fields['inner']['seconds'], full_node_count=m.n_material)
            saved = dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H), H_used=np.asarray(H),
                damage_image=np.asarray(fields['damage_image']), q_previous=np.asarray(fields['q_previous']),
                d_previous=np.asarray(fields['d_previous']), H_parent=np.asarray(parent_H), d_parent=np.asarray(parent_d))
            if index == 1 and sweep == 1:
                atomic_json(root/'preflight.json', dict(full_loss_CSR_parity=True,
                    gpu_step_compiled=True, own_zero=True, common_initialization_checked=True))
            atomic_json(folder/f'sweep_{sweep:04d}.json', row)
            dest = folder/f'outer_{sweep:04d}' if sweep in a.checkpoint_sweeps or stop['converged'] else folder/'pending'
            checkpoint(dest, saved, dict(row), carry)
            atomic_json(root/'result.json', dict(complete=False, current=row, curve=curve,
                case=a.case, requested_loads=grid.tolist(), optimizer_count=count))
            if sweep == 1 or sweep % 8 == 0:
                print(json.dumps(dict(event='cold_full_outer', **row)), flush=True)
            if stop['converged']: break
        q_sync, reaction, mechanical = solve(d, load, q)
        H_sync = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q_sync))))
        force = m.matrix(np.asarray(d))@np.asarray(q_sync)
        norm = float(np.linalg.norm(force[m.free]))
        if norm > mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('final independent host FEM force check fails')
        np.testing.assert_allclose(force@m.load_direction, reaction, rtol=1e-9, atol=1e-6)
        physical = residual_metrics(np.asarray(d), np.asarray(H_sync), D, gc0=g0, gc1=g1)[1]
        saved.update(q_stage=np.asarray(q), q=np.asarray(q_sync), H=np.asarray(H_sync),
            raw_current_image=np.asarray(image(carry[0], context(H_sync))))
        report = dict(row, reaction=float(reaction), q_d_synchronized=True,
            mechanics_calls=sweep+1, mechanics=mechanical, host_force_norm=norm,
            physical_AT2=physical, original_residual_pass=bool(physical<=c.reporting.coupled_AT2_relative_limit),
            certified_path=False, status='provisional_field_stable' if stop['converged'] else 'provisional_budget_exhausted',
            seconds=time.perf_counter()-load_started,
            crack=crack_extent(np.asarray(d), m.coordinates[:m.n_material], adjacency,
                threshold=.9, tip=[.5, .5], tip_radius=c.physical.l0, right_edge=1., tolerance=1e-12))
        checkpoint(folder, saved, report, carry)
        atomic_json(folder/'result.json', report)
        curve.append(report); q, H = q_sync, H_sync
        atomic_json(root/'result.json', dict(complete=False, current=report, curve=curve,
            case=a.case, requested_loads=grid.tolist(), optimizer_count=count))
        print(json.dumps(dict(event='cold_full_load_complete', **report)), flush=True)
    for path, digest in protected.items():
        if sha(path) != digest: raise RuntimeError(f'physical input changed: {path}')
    for name, digest in code.items():
        if sha(repo/name) != digest: raise RuntimeError(f'numerical source changed: {name}')
    atomic_json(root/'result.json', dict(complete=True, case=a.case, curve=curve, current=curve[-1],
        requested_loads=grid.tolist(), optimizer_count=count, seconds=time.perf_counter()-started,
        certified_path=False, own_zero=True, sampling='all_nodes', loss_norm='l1'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(runpy.run_path(args.config)['get_config'](), args.config)
