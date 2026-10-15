"""Common-parent mechanics-first direct/PINN fixed-load stagger controls."""
import argparse
import hashlib
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
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d._legacy.at2_cg import damage_solver
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.core_halo_damage import pack_rows, validate_probabilities
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.l1_damage_loading import residual_metrics, committed_history
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import make_loss_fn, make_optimizer, make_step, optimizer_count
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.prebuilt_damage_pool import build_sampling_pool, make_pool_objective
from examples.fracture2d._legacy.precomputed_damage_sampling import make_sampling_rows
from examples.fracture2d._legacy.provisional_damage import mesh_adjacency, crack_extent
from examples.fracture2d._legacy.staggered_damage import case_configs, mechanics_first_sweep, field_changes, field_stop


def read_fields(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def prepare(a):
    case_configs(a.cases)
    if (a.load <= a.parent_load or a.final_mechanics_synchronization is not True
            or a.advance_on_budget_exit is not False or a.parent_count < 1
            or type(a.cpu_threads) is not int or a.cpu_threads < 1):
        raise ValueError('control requires a positive fixed load, final sync and no budget advancement')
    root = Path(a.source_root)
    for name, digest in a.pins.items():
        if sha(root/name) != digest:
            raise ValueError(f'pinned control source changed: {name}')
    identity = json.loads((root/'identity.json').read_text())
    repo = Path(__file__).resolve().parents[3]
    for name, digest in identity['code'].items():
        if sha(repo/name) != digest:
            raise ValueError(f'protected source changed: {name}')
    c = ConfigDict(identity['config'], allow_dotted_keys=True)
    parent = root/f'state_{a.parent_index:04d}'
    report = json.loads((parent/'result.json').read_text())
    audit = json.loads((root/'independent_review/audit.json').read_text())
    matches = [r for r in audit['rows'] if r['folder'] == parent.name and r['final_load']]
    if (report['load'] != a.parent_load or report['optimizer_count'] != a.parent_count
            or len(matches) != 1 or not matches[0]['pass_replay']):
        raise ValueError('control parent load/count/independent replay differs')
    m, op, geometry = load_geometry(c)
    if geometry != identity['geometry']:
        raise ValueError('control geometry differs from parent')
    B, D = load_npz(c.l1_loading.basis_path), op.to_scipy()
    if sha(c.l1_loading.basis_path) != c.l1_loading.basis_sha256:
        raise ValueError('parent basis changed')
    parameterization = read_fields(root/'parameterization.npz')
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    features, info = make_features(op, m.coordinates[:m.n_material], gc0=g0,
        length=c.physical.l0, tip=list(c.conditioned.tip), powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    if info != identity['features']:
        raise ValueError('parent feature definition differs')
    features = eqx.filter_jit(features)
    net_template = fresh_network(c, info['input_dimension'], B, head='sigmoid')
    t = c.l1_loading
    optimizer, schedule = make_optimizer(kind=t.optimizer, learning_rate=t.learning_rate,
        decay_steps=t.decay_steps, decay_rate=t.decay_rate, minimum_rate=t.minimum_rate,
        grad_clip=t.grad_clip, b1=t.b1, b2=t.b2, eps=t.eps, eps_root=t.eps_root,
        shampoo_beta=t.shampoo_beta, correct_bias=t.correct_bias,
        precondition_frequency=t.precondition_frequency, max_precond_dim=t.max_precond_dim,
        precondition_1d=t.precondition_1d, weight_decay=t.weight_decay)
    state_template = optimizer.init(eqx.filter(net_template, eqx.is_inexact_array))
    net = eqx.tree_deserialise_leaves(parent/'network.eqx', net_template)
    state = eqx.tree_deserialise_leaves(parent/'optimizer.eqx', state_template)
    if optimizer_count(state, kind='adam') != a.parent_count:
        raise ValueError('restored parent Adam count differs')
    bc, bv = pack_rows(B@diags(parameterization['column_scale']), width=int(np.diff(B.indptr).max()))
    dc, dv = pack_rows(D, width=int(np.diff(D.indptr).max()))
    bc, bv, dc, dv = map(jnp.asarray, (bc, bv, dc, dv))
    image, objective = make_loss_fn(gc0=g0, gc1=g1, gain=float(parameterization['trunk_gain']),
        population_size=m.n_material, norm=t.loss_norm)
    image = eqx.filter_jit(image)
    step = make_step(make_pool_objective(objective, basis_columns=bc, basis_values=bv), optimizer)
    def full(H):
        return dict(H=H, core_H=H, features=jax.lax.stop_gradient(features(H)),
            basis_columns=bc, basis_values=bv, core_local=jnp.arange(m.n_material, dtype=jnp.int32),
            operator_columns=dc, operator_values=dv,
            p_core=jnp.full(m.n_material, 1/m.n_material, jnp.float64))
    probabilities = np.load(root/'sampling_probabilities.npy')
    validate_probabilities(probabilities, population_size=m.n_material)
    rows = make_sampling_rows(D, row_width=dc.shape[1])
    if a.parent_count % t.pool_size:
        raise ValueError('control parent must end exactly at a sampling-block boundary')
    blocks = (root/'sampling_blocks.jsonl').read_text().splitlines()
    block = json.loads(blocks[a.parent_count//t.pool_size-1])
    if block['block'] != a.parent_count//t.pool_size:
        raise ValueError('parent sampling journal/count differs')
    return dict(c=c, parent=parent, report=report, identity=identity, m=m, op=op,
        D=D, B=B, g0=g0, g1=g1, features=features, image=image, step=step, full=full,
        schedule=schedule, net=net, state=state, probabilities=probabilities, rows=rows,
        rng_state=block['rng_after'], fields=read_fields(parent/'fields.npz'))


def run(a, config_path, *, preflight_only):
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise ValueError('stagger control requires serialized FP64 GPU execution')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    p = prepare(a)
    m, c, D, op, f = (p[k] for k in ['m', 'c', 'D', 'op', 'fields'])
    t = c.l1_loading
    solve = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
        maxiter=c.validation.cg_maxiter)
    solve_d = damage_solver(op, m.volumes, p['g0'], p['g1'], tol=c.validation.cg_rtol,
        atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)
    # Compilation does not consume parent optimizer state/RNG or case counters.
    q0, _, _ = solve(jnp.asarray(f['d']), a.load, jnp.asarray(f['q']))
    h0 = committed_history(jnp.asarray(f['H']), jnp.asarray(m.driver(np.asarray(q0))))
    restored = p['image'](p['net'], p['full'](jnp.asarray(f['H_used'])))
    np.testing.assert_allclose(restored, f['damage_image'], rtol=1e-10, atol=1e-12)
    rng = np.random.default_rng(); rng.bit_generator.state = p['rng_state']
    cores = np.stack([rng.choice(m.n_material, size=t.sample_count, replace=True,
        p=p['probabilities']).astype(np.int32) for _ in range(t.pool_size)])
    pool, _ = build_sampling_pool(p['rows'], cores, p['probabilities'], node_capacity=t.halo_capacity)
    warm = dict(H=h0, features=jax.lax.stop_gradient(p['features'](h0)),
        pool=jax.device_put(pool), slot=jnp.asarray(0, jnp.int32))
    jax.block_until_ready(p['step'](p['net'], p['state'], warm, jnp.asarray(a.load)))
    solve_d(h0, jnp.asarray(f['d'])).block_until_ready()
    if preflight_only:
        print(json.dumps(dict(preflight=True, parent_count=a.parent_count,
            parent_image_parity=True, gpu_step_compiled=True)), flush=True)
        return
    root = Path(a.output_dir)
    if (root/'identity.json').exists():
        raise ValueError('refusing to overwrite existing stagger control')
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    code = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):
                code[str(path.relative_to(repo))] = sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))] = sha(config_path)
    for name in code:
        target = root/'code_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(repo/name, target)
    atomic_json(root/'identity.json', dict(config=a.to_dict(), source_config=c.to_dict(),
        geometry=p['identity']['geometry'], code=code, parent_certified_path=p['report']['certified_path'],
        model_unchanged=True, reference_training_targets=False,
        order='mechanics -> committed-floor H -> frozen-H damage -> field-change check',
        outer_mechanics_calls=1, final_synchronization_calls=1, rng_state=p['rng_state']))
    atomic_json(root/'preflight.json', dict(pass_restore=True, parent_image_parity=True,
        gpu_step_compiled=True, optimizer_count=a.parent_count))
    summaries = []
    adjacency = mesh_adjacency(m.triangles[:m.n_material_cells], population_size=m.n_material)
    material_triangles, material_areas = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    for case in case_configs(a.cases):
        if (case.solver not in ['direct', 'pinn'] or not 0 < case.omega <= 1
                or type(case.max_sweeps) is not int or case.max_sweeps < a.minimum_sweeps):
            raise ValueError('invalid explicit stagger case')
        folder = root/case.name; folder.mkdir()
        q, d, parent_H = map(jnp.asarray, (f['q'], f['d'], f['H']))
        carry = (p['net'], p['state'])
        rng = np.random.default_rng(); rng.bit_generator.state = p['rng_state']
        cache = dict(slot=t.pool_size, block=0, pool=None)
        def sampled(H, X):
            if cache['slot'] == t.pool_size:
                before = rng.bit_generator.state
                cores = np.stack([rng.choice(m.n_material, size=t.sample_count, replace=True,
                    p=p['probabilities']).astype(np.int32) for _ in range(t.pool_size)])
                pool, halos = build_sampling_pool(p['rows'], cores, p['probabilities'], node_capacity=t.halo_capacity)
                cache.update(slot=0, block=cache['block']+1, pool=jax.device_put(pool))
                with (folder/'sampling_blocks.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(block=cache['block'], rng_before=before,
                        rng_after=rng.bit_generator.state, cores_sha256=hashlib.sha256(cores.tobytes()).hexdigest(),
                        halo_min=int(halos.min()), halo_max=int(halos.max())))+'\n')
            slot = cache['slot']; cache['slot'] += 1
            return dict(H=H, features=X, pool=cache['pool'], slot=jnp.asarray(slot, jnp.int32))
        def damage(H, old_d, previous):
            start = time.perf_counter()
            if case.solver == 'direct':
                image = solve_d(H, old_d); new_carry = previous
            else:
                net, state = previous; X = jax.lax.stop_gradient(p['features'](H))
                for _ in range(t.inner_updates):
                    net, state, value, gradient = p['step'](net, state, sampled(H, X), jnp.asarray(a.load))
                if not np.isfinite([float(value), float(gradient)]).all():
                    raise FloatingPointError('neural stagger objective/gradient is nonfinite')
                image = p['image'](net, p['full'](H)); new_carry = (net, state)
            image.block_until_ready()
            m.factors(np.asarray(image))
            inner = residual_metrics(np.asarray(image), np.asarray(H), D, gc0=p['g0'], gc1=p['g1'])[1]
            return image, new_carry, dict(AT2_relative=inner, seconds=time.perf_counter()-start)
        counter = loose_counter = 0; loose_first = None; trace = []
        started = time.perf_counter()
        def checkpoint(destination, fields, row, state):
            destination.mkdir(exist_ok=True)
            atomic_fields(destination/'fields.npz', **fields)
            row['fields_sha256'] = sha(destination/'fields.npz')
            if case.solver == 'pinn':
                eqx.tree_serialise_leaves(destination/'network.eqx', state[0])
                eqx.tree_serialise_leaves(destination/'optimizer.eqx', state[1])
                row.update(network_sha256=sha(destination/'network.eqx'), optimizer_sha256=sha(destination/'optimizer.eqx'))
            atomic_json(destination/'report.json', row)
        for sweep in range(1, case.max_sweeps+1):
            step_fields, carry = mechanics_first_sweep(q, d, parent_H, a.load, carry,
                mechanics=solve, driver=m.driver, damage=damage, omega=case.omega)
            q, d, H = (step_fields[k] for k in ['q', 'd', 'H'])
            changes = field_changes(q, step_fields['q_previous'], d, step_fields['d_previous'],
                material_triangles, material_areas, load=a.load)
            stopping = field_stop(changes, displacement_limit=a.displacement_change_limit,
                damage_limit=a.damage_change_limit, sweep=sweep, minimum_sweeps=a.minimum_sweeps,
                consecutive_passes=a.consecutive_passes, previous_consecutive=counter)
            loose = field_stop(changes, displacement_limit=a.loose_displacement_limit,
                damage_limit=a.loose_damage_limit, sweep=sweep, minimum_sweeps=a.minimum_sweeps,
                consecutive_passes=a.consecutive_passes, previous_consecutive=loose_counter)
            counter, loose_counter = stopping['consecutive'], loose['consecutive']
            if loose['converged'] and loose_first is None: loose_first = sweep
            count = optimizer_count(carry[1], kind='adam')
            expected = a.parent_count + (sweep*t.inner_updates if case.solver == 'pinn' else 0)
            if count != expected: raise RuntimeError('Adam count reset across stagger sweeps')
            row = dict(case=case.name, solver=case.solver, omega=case.omega, sweep=sweep,
                load=a.load, **changes, field_stopping=stopping, loose_first_stable=loose_first,
                mechanics_calls=sweep, reaction=step_fields['reaction'], reaction_damage='d_previous',
                q_d_synchronized=False, inner_AT2=step_fields['inner']['AT2_relative'],
                stage_AT2=residual_metrics(np.asarray(d), np.asarray(H), D, gc0=p['g0'], gc1=p['g1'])[1],
                mechanics=step_fields['mechanics'], damage_max=float(d.max()), optimizer_count=count,
                learning_rate=float(p['schedule'](count-1)), seconds=time.perf_counter()-started)
            saved = dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H), H_used=np.asarray(H),
                damage_image=np.asarray(step_fields['damage_image']), q_previous=np.asarray(step_fields['q_previous']),
                d_previous=np.asarray(step_fields['d_previous']), H_parent=f['H'], d_parent=f['d'])
            atomic_json(folder/f'sweep_{sweep:04d}.json', row)
            if sweep in a.checkpoints or stopping['converged'] or sweep == case.max_sweeps:
                checkpoint(folder/f'outer_{sweep:04d}', saved, dict(row), carry)
            trace.append(row)
            atomic_json(root/'result.json', dict(complete=False, current=row, cases=summaries))
            if sweep == 1 or sweep % 8 == 0:
                print(json.dumps(dict(event='stagger_sweep', **row)), flush=True)
            if stopping['converged']: break
        # One endpoint synchronization; no history commitment or load advancement.
        q_sync, reaction, mechanical = solve(d, a.load, q)
        H_sync = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q_sync))))
        physical = residual_metrics(np.asarray(d), np.asarray(H_sync), D, gc0=p['g0'], gc1=p['g1'])[1]
        force = m.matrix(np.asarray(d))@np.asarray(q_sync)
        norm = float(np.linalg.norm(force[m.free]))
        if norm > mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('endpoint independent host FEM force check fails')
        np.testing.assert_allclose(force@m.load_direction, reaction, rtol=1e-9, atol=1e-6)
        saved.update(q_stage=saved['q'], q=np.asarray(q_sync), H=np.asarray(H_sync))
        if case.solver == 'pinn':
            saved['raw_current_image'] = np.asarray(p['image'](carry[0], p['full'](H_sync)))
        final = dict(row, reaction=float(reaction), mechanics=mechanical, host_force_norm=norm,
            mechanics_calls=sweep+1, final_sync_calls=1, q_d_synchronized=True,
            physical_AT2=physical, original_residual_pass=bool(physical <= c.reporting.coupled_AT2_relative_limit),
            field_stable=stopping['converged'], certified_path=False, load_advanced=False,
            status='field_stable' if stopping['converged'] else 'field_change_budget_exhausted',
            crack=crack_extent(np.asarray(d), m.coordinates[:m.n_material], adjacency,
                threshold=.9, tip=[.5, .5], tip_radius=c.physical.l0, right_edge=1., tolerance=1e-12),
            seconds=time.perf_counter()-started)
        checkpoint(folder/'final', saved, final, carry)
        atomic_json(folder/'result.json', final)
        summaries.append(final)
        print(json.dumps(dict(event='stagger_case_complete', **final)), flush=True)
    for name, digest in a.pins.items():
        if sha(Path(a.source_root)/name) != digest: raise RuntimeError('protected input changed during trial')
    for name, digest in code.items():
        if sha(repo/name) != digest: raise RuntimeError('numerical source changed during trial')
    atomic_json(root/'result.json', dict(complete=True, cases=summaries, current=summaries[-1],
        model_unchanged=True, fixed_load=a.load, load_advanced=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    run(runpy.run_path(args.config)['get_config'](), args.config, preflight_only=args.preflight_only)
