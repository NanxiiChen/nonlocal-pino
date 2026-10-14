"""Fixed-load equal-outer-count PINN controls with exact stage restarts."""
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
import numpy as np

from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.l1_damage_loading import residual_metrics, committed_history
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.low_order_damage import optimizer_count
from examples.fracture2d.output_damage_continue import atomic_fields
from examples.fracture2d.prebuilt_damage_pool import build_sampling_pool
from examples.fracture2d.provisional_damage import mesh_adjacency, crack_extent
from examples.fracture2d.staggered_damage import (
    case_configs, mechanics_first_sweep, field_changes, field_stop)
from examples.fracture2d.staggered_damage_trial import prepare, read_fields


def restart_plan(a, source_inner_updates, pool_size):
    """Require complete sampling blocks and an unchanged baseline optimizer."""
    case, = case_configs(a.cases)
    if (case.solver != 'pinn' or case.omega != 1. or a.stop_early is not False
            or type(a.start_sweep) is not int or not 0 <= a.start_sweep < case.max_sweeps
            or type(a.inner_updates) is not int or a.inner_updates < 1
            or (case.max_sweeps-a.start_sweep)*a.inner_updates % pool_size):
        raise ValueError('equal-count control requires omega1 and complete sampling blocks')
    if a.start_sweep:
        if (not a.restart_root or not a.restart_pins
                or a.inner_updates != source_inner_updates
                or a.start_sweep*source_inner_updates % pool_size):
            raise ValueError('continued baseline requires pinned stage and unchanged inner budget')
    elif a.restart_root is not None or len(a.restart_pins):
        raise ValueError('fresh control must use only the preceding-load parent')
    return dict(case=case, initial_count=a.parent_count+a.start_sweep*source_inner_updates,
                updates=(case.max_sweeps-a.start_sweep)*a.inner_updates)


def restore(a, p):
    plan = restart_plan(a, p['c'].l1_loading.inner_updates, p['c'].l1_loading.pool_size)
    f = p['fields']
    if a.start_sweep:
        root = Path(a.restart_root)
        for name, digest in a.restart_pins.items():
            if sha(root/name) != digest:
                raise ValueError(f'continuation input changed: {name}')
        old = json.loads((root/'identity.json').read_text())
        if (old['config']['source_root'] != a.source_root
                or old['config']['pins'] != a.pins.to_dict()
                or old['config']['load'] != a.load):
            raise ValueError('continuation parent/model/load differs')
        for name, digest in old['code'].items():
            if sha(Path(__file__).resolve().parents[2]/name) != digest:
                raise ValueError(f'previous stagger source changed: {name}')
        folder = root/'pinn_full'/f'outer_{a.start_sweep:04d}'
        report = json.loads((folder/'report.json').read_text())
        audit = json.loads((root/'independent_review/audit.json').read_text())
        matches = [r for r in audit['rows'] if r['case'] == 'pinn_full'
                   and r['sweep'] == a.start_sweep and not r['final']]
        if (not audit['complete'] or len(matches) != 1 or not matches[0]['pass_replay']
                or report['optimizer_count'] != plan['initial_count']
                or report['omega'] != 1. or report['load'] != a.load
                or report['q_d_synchronized'] is not False):
            raise ValueError('continuation requires independently audited unrelaxed stage')
        stage = read_fields(folder/'fields.npz')
        np.testing.assert_array_equal(stage['H_parent'], f['H'])
        np.testing.assert_array_equal(stage['d_parent'], f['d'])
        p['net'] = eqx.tree_deserialise_leaves(folder/'network.eqx', p['net'])
        p['state'] = eqx.tree_deserialise_leaves(folder/'optimizer.eqx', p['state'])
        blocks = [json.loads(line) for line in (root/'pinn_full/sampling_blocks.jsonl').read_text().splitlines()]
        if len(blocks) != a.start_sweep*p['c'].l1_loading.inner_updates//p['c'].l1_loading.pool_size:
            raise ValueError('restart stage does not end at archived RNG boundary')
        p['rng_state'] = blocks[-1]['rng_after']
        initial = stage
    else:
        initial = f
        report = p['report']
    if optimizer_count(p['state'], kind='adam') != plan['initial_count']:
        raise ValueError('restart Adam state/count differs')
    return plan, initial, report


def run(a, config_path, *, preflight_only):
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise ValueError('continuation requires serialized FP64 GPU execution')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    p = prepare(a)
    plan, initial, prior = restore(a, p)
    case = plan['case']
    m, c, D, f = (p[k] for k in ['m', 'c', 'D', 'fields'])
    # Inner count does not enter loss/optimizer construction; declare the branch budget.
    c.l1_loading.inner_updates = a.inner_updates
    t = c.l1_loading
    solve = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                         maxiter=c.validation.cg_maxiter)
    q, d, parent_H = map(jnp.asarray, (initial['q'], initial['d'], f['H']))
    H = jnp.asarray(initial['H_used'])
    image = p['image'](p['net'], p['full'](H))
    np.testing.assert_allclose(image, initial['damage_image'], rtol=1e-10, atol=1e-12)
    # Discard preflight computations without consuming persistent optimizer/RNG.
    q_warm, _, _ = solve(d, a.load, q)
    H_warm = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q_warm))))
    rng = np.random.default_rng(); rng.bit_generator.state = p['rng_state']
    cores = np.stack([rng.choice(m.n_material, size=t.sample_count, replace=True,
                     p=p['probabilities']).astype(np.int32) for _ in range(t.pool_size)])
    pool, _ = build_sampling_pool(p['rows'], cores, p['probabilities'], node_capacity=t.halo_capacity)
    warm = dict(H=H_warm, features=jax.lax.stop_gradient(p['features'](H_warm)),
                pool=jax.device_put(pool), slot=jnp.asarray(0, jnp.int32))
    jax.block_until_ready(p['step'](p['net'], p['state'], warm, jnp.asarray(a.load)))
    if preflight_only:
        print(json.dumps(dict(pass_restore=True, parent_image_parity=True,
                              initial_count=plan['initial_count'], start_sweep=a.start_sweep)), flush=True)
        return
    root = Path(a.output_dir)
    if (root/'identity.json').exists():
        raise ValueError('refusing to overwrite a continuation')
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
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
    atomic_json(root/'identity.json', dict(config=a.to_dict(), source_config=c.to_dict(), code=code,
        geometry=p['identity']['geometry'], rng_state=p['rng_state'], initial_count=plan['initial_count'],
        preceding_load_history_unchanged=True, reference_training_targets=False,
        parent_certified_path=p['report']['certified_path'], model_unchanged=True,
        outer_mechanics_calls=1, final_synchronization_calls=1,
        omitted_from_trajectory_prior_endpoint_sync_calls=int(a.start_sweep > 0),
        preflight_mechanics_calls=1))
    atomic_json(root/'preflight.json', dict(pass_restore=True, parent_image_parity=True,
        gpu_step_compiled=True, optimizer_count=plan['initial_count'], start_sweep=a.start_sweep))
    folder = root/case.name; folder.mkdir()
    rng.bit_generator.state = p['rng_state']
    cache = dict(slot=t.pool_size, block=0, pool=None)
    carry = (p['net'], p['state'])
    counter = prior['field_stopping']['consecutive'] if a.start_sweep else 0
    loose_counter = 0
    loose_first = prior['loose_first_stable'] if a.start_sweep else None
    adjacency = mesh_adjacency(m.triangles[:m.n_material_cells], population_size=m.n_material)
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    started = time.perf_counter()

    def damage(H, old_d, previous):
        net, state = previous; X = jax.lax.stop_gradient(p['features'](H))
        for _ in range(t.inner_updates):
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
            batch = dict(H=H, features=X, pool=cache['pool'], slot=jnp.asarray(cache['slot'], jnp.int32))
            cache['slot'] += 1
            net, state, value, gradient = p['step'](net, state, batch, jnp.asarray(a.load))
        if not np.isfinite([float(value), float(gradient)]).all():
            raise FloatingPointError('nonfinite continuation objective/gradient')
        image = p['image'](net, p['full'](H)); image.block_until_ready()
        m.factors(np.asarray(image))
        inner = residual_metrics(np.asarray(image), np.asarray(H), D, gc0=p['g0'], gc1=p['g1'])[1]
        return image, (net, state), dict(AT2_relative=inner)

    def checkpoint(destination, fields, row):
        destination.mkdir(exist_ok=True)
        atomic_fields(destination/'fields.npz', **fields)
        eqx.tree_serialise_leaves(destination/'network.eqx', carry[0])
        eqx.tree_serialise_leaves(destination/'optimizer.eqx', carry[1])
        row.update(fields_sha256=sha(destination/'fields.npz'),
            network_sha256=sha(destination/'network.eqx'), optimizer_sha256=sha(destination/'optimizer.eqx'))
        atomic_json(destination/'report.json', row)

    for sweep in range(a.start_sweep+1, case.max_sweeps+1):
        fields, carry = mechanics_first_sweep(q, d, parent_H, a.load, carry,
            mechanics=solve, driver=m.driver, damage=damage, omega=1.)
        q, d, H = (fields[k] for k in ['q', 'd', 'H'])
        changes = field_changes(q, fields['q_previous'], d, fields['d_previous'], tri, area, load=a.load)
        stopping = field_stop(changes, displacement_limit=a.displacement_change_limit,
            damage_limit=a.damage_change_limit, sweep=sweep, minimum_sweeps=a.minimum_sweeps,
            consecutive_passes=a.consecutive_passes, previous_consecutive=counter)
        loose = field_stop(changes, displacement_limit=a.loose_displacement_limit,
            damage_limit=a.loose_damage_limit, sweep=sweep, minimum_sweeps=a.minimum_sweeps,
            consecutive_passes=a.consecutive_passes, previous_consecutive=loose_counter)
        counter, loose_counter = stopping['consecutive'], loose['consecutive']
        if loose['converged'] and loose_first is None: loose_first = sweep
        count = optimizer_count(carry[1], kind='adam')
        if count != plan['initial_count']+(sweep-a.start_sweep)*t.inner_updates:
            raise RuntimeError('continuation Adam count reset')
        row = dict(case=case.name, solver='pinn', omega=1., sweep=sweep, load=a.load,
            **changes, field_stopping=stopping, loose_first_stable=loose_first,
            mechanics_calls=sweep, new_mechanics_calls=sweep-a.start_sweep,
            reaction=fields['reaction'], reaction_damage='d_previous', q_d_synchronized=False,
            inner_AT2=fields['inner']['AT2_relative'], stage_AT2=fields['inner']['AT2_relative'],
            mechanics=fields['mechanics'], damage_max=float(d.max()), optimizer_count=count,
            learning_rate=float(p['schedule'](count-1)), seconds=time.perf_counter()-started)
        saved = dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H), H_used=np.asarray(H),
            damage_image=np.asarray(fields['damage_image']), q_previous=np.asarray(fields['q_previous']),
            d_previous=np.asarray(fields['d_previous']), H_parent=f['H'], d_parent=f['d'])
        atomic_json(folder/f'sweep_{sweep:04d}.json', row)
        if sweep in a.checkpoints or sweep == case.max_sweeps:
            checkpoint(folder/f'outer_{sweep:04d}', saved, dict(row))
        atomic_json(root/'result.json', dict(complete=False, current=row, cases=[]))
        if sweep == a.start_sweep+1 or sweep % 8 == 0:
            print(json.dumps(dict(event='stagger_continue_sweep', **row)), flush=True)
    q_sync, reaction, mechanical = solve(d, a.load, q)
    H_sync = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q_sync))))
    physical = residual_metrics(np.asarray(d), np.asarray(H_sync), D, gc0=p['g0'], gc1=p['g1'])[1]
    force = m.matrix(np.asarray(d))@np.asarray(q_sync)
    norm = float(np.linalg.norm(force[m.free]))
    if norm > mechanical['convergence_limit']*(1+1e-3):
        raise RuntimeError('continuation endpoint host FEM force check fails')
    np.testing.assert_allclose(force@m.load_direction, reaction, rtol=1e-9, atol=1e-6)
    saved.update(q_stage=saved['q'], q=np.asarray(q_sync), H=np.asarray(H_sync),
                 raw_current_image=np.asarray(p['image'](carry[0], p['full'](H_sync))))
    final = dict(row, reaction=float(reaction), reaction_damage='d', mechanics=mechanical,
        host_force_norm=norm, mechanics_calls=sweep+1, new_mechanics_calls=sweep-a.start_sweep+1,
        final_sync_calls=1, q_d_synchronized=True, physical_AT2=physical,
        original_residual_pass=bool(physical <= c.reporting.coupled_AT2_relative_limit),
        field_stable=stopping['converged'], certified_path=False, load_advanced=False,
        status='matched_direct_outer_budget_complete', seconds=time.perf_counter()-started,
        crack=crack_extent(np.asarray(d), m.coordinates[:m.n_material], adjacency,
            threshold=.9, tip=[.5, .5], tip_radius=c.physical.l0, right_edge=1., tolerance=1e-12))
    checkpoint(folder/'final', saved, final)
    atomic_json(folder/'result.json', final)
    for source, pins in [(a.source_root, a.pins), (a.restart_root, a.restart_pins)]:
        for name, digest in pins.items():
            if sha(Path(source)/name) != digest: raise RuntimeError('protected continuation input changed')
    for name, digest in code.items():
        if sha(repo/name) != digest: raise RuntimeError('continuation numerical source changed')
    atomic_json(root/'result.json', dict(complete=True, cases=[final], current=final,
        fixed_load=a.load, load_advanced=False, model_unchanged=True))
    print(json.dumps(dict(event='stagger_continue_complete', **final)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    run(runpy.run_path(args.config)['get_config'](), args.config, preflight_only=args.preflight_only)
