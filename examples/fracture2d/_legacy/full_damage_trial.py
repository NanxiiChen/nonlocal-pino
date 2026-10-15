"""All-node L1 damage: same-H optimizer screen, then fixed-load stagger."""
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

from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config, make_lbfgs, choose_optimizer
from examples.fracture2d._legacy.l1_damage_loading import residual_metrics, committed_history
from examples.fracture2d._legacy.local_damage_trial import sha
from examples.fracture2d._legacy.low_order_damage import make_loss_fn, make_step, optimizer_count
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.provisional_damage import mesh_adjacency, crack_extent
from examples.fracture2d._legacy.staggered_damage import case_configs, mechanics_first_sweep, field_changes, field_stop
from examples.fracture2d._legacy.staggered_damage_trial import prepare


def run(a, config_path, *, preflight_only):
    t = a.full_training
    case, = case_configs(a.cases)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or t.sampling != 'all_nodes' or t.norm != 'l1' or t.stop_early is not False
            or case.solver != 'pinn' or case.omega != 1. or t.inner_updates != t.lbfgs.maxiter
            or t.pilot_updates != t.inner_updates or t.selection != 'minimum_full_loss'
            or t.adam_memory != 'parent_persistent' or type(t.inner_updates) is not int or t.inner_updates < 1):
        raise ValueError('full-domain control requires explicit same-loss fixed-load GPU protocol')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    p = prepare(a)
    m, c, D, f = (p[k] for k in ['m', 'c', 'D', 'fields'])
    image, objective = make_loss_fn(gc0=p['g0'], gc1=p['g1'],
        gain=float(np.load(Path(a.source_root)/'parameterization.npz')['trunk_gain']),
        population_size=m.n_material, norm=t.norm)
    image, value = eqx.filter_jit(image), eqx.filter_jit(objective)
    optimizer, schedule = adam_from_config(c.l1_loading)
    adam_step = make_step(objective, optimizer)
    fit, lbfgs_value_grad = make_lbfgs(p['net'], objective, t.lbfgs)
    solve = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                         maxiter=c.validation.cg_maxiter)
    q0, reaction0, mechanical0 = solve(jnp.asarray(f['d']), a.load, jnp.asarray(f['q']))
    H0 = committed_history(jnp.asarray(f['H']), jnp.asarray(m.driver(np.asarray(q0))))
    context0 = p['full'](H0)
    restored = image(p['net'], p['full'](jnp.asarray(f['H_used'])))
    np.testing.assert_allclose(restored, f['damage_image'], rtol=1e-10, atol=1e-12)
    # Actual all-row operator/loss parity; no sampled support or reference target.
    d0 = np.asarray(image(p['net'], context0))
    raw = (p['g0']+2*np.asarray(H0))*d0+p['g1']*(D@d0)-2*np.asarray(H0)
    expected = np.mean(abs(raw))/(2*np.sqrt(np.mean(np.asarray(H0)**2)))
    np.testing.assert_allclose(value(p['net'], context0, jnp.asarray(a.load)), expected, rtol=1e-10, atol=1e-12)
    jax.block_until_ready(adam_step(p['net'], p['state'], context0, jnp.asarray(a.load)))
    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(eqx.filter(p['net'], eqx.is_inexact_array))
    jax.block_until_ready(lbfgs_value_grad(flat, context0, jnp.asarray(a.load)))
    if preflight_only:
        print(json.dumps(dict(pass_restore=True, all_nodes=m.n_material, full_loss_parity=True)), flush=True)
        return
    root = Path(a.output_dir)
    if (root/'identity.json').exists(): raise ValueError('refusing existing full-domain trial')
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
    atomic_json(root/'identity.json', dict(config=a.to_dict(), source_config=c.to_dict(), code=code,
        geometry=p['identity']['geometry'], model_unchanged=True, full_node_count=m.n_material,
        sampling='all_nodes', loss_norm=t.norm, reference_training_targets=False,
        history_floor_load=a.parent_load, preflight_mechanics_calls=1))
    atomic_json(root/'preflight.json', dict(pass_restore=True, parent_image_parity=True,
        full_loss_parity=True, gpu_step_compiled=True, full_node_count=m.n_material))

    def train(kind, H, previous):
        context = p['full'](H)
        net, state = previous
        if kind == 'lbfgs':
            net, report = fit(net, context, jnp.asarray(a.load))
        elif kind == 'adam':
            started = time.perf_counter()
            for _ in range(t.inner_updates):
                net, state, loss, gradient = adam_step(net, state, context, jnp.asarray(a.load))
            if not np.isfinite([float(loss), float(gradient)]).all():
                raise FloatingPointError('nonfinite all-node Adam objective/gradient')
            report = dict(optimizer='adam', iterations=t.inner_updates, evaluations=t.inner_updates,
                optimizer_success=True, status='declared_budget', message='persistent full-domain Adam',
                seconds=time.perf_counter()-started)
        else: raise ValueError('unknown selected full-domain optimizer')
        d = image(net, context); d.block_until_ready(); m.factors(np.asarray(d))
        report.update(full_loss=float(value(net, context, jnp.asarray(a.load))),
            AT2_relative=residual_metrics(np.asarray(d), np.asarray(H), D, gc0=p['g0'], gc1=p['g1'])[1])
        return d, (net, state), report

    def checkpoint(folder, fields, row, carry, kind):
        folder.mkdir(parents=True, exist_ok=True)
        atomic_fields(folder/'fields.npz', **fields)
        eqx.tree_serialise_leaves(folder/'network.eqx', carry[0])
        row.update(fields_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'))
        if kind == 'adam':
            eqx.tree_serialise_leaves(folder/'optimizer.eqx', carry[1])
            row.update(optimizer_sha256=sha(folder/'optimizer.eqx'), optimizer_count=optimizer_count(carry[1], kind='adam'))
        else: row['optimizer_count'] = None
        atomic_json(folder/'report.json', row)

    candidates = []
    for kind in ['adam', 'lbfgs']:
        d, memory, report = train(kind, H0, (p['net'], p['state']))
        row = dict(report, physical_state_promoted=False, parent_optimizer_count=a.parent_count,
            load=a.load, full_node_count=m.n_material, reaction_at_parent_damage=float(reaction0),
            mechanics=mechanical0)
        checkpoint(root/'screen'/kind, dict(damage_image=np.asarray(d), H=np.asarray(H0),
            q_mechanics=np.asarray(q0), H_parent=f['H'], d_parent=f['d']), row, memory, kind)
        candidates.append(row)
        print(json.dumps(dict(event='full_domain_screen', **row)), flush=True)
    selected = choose_optimizer(candidates)
    atomic_json(root/'selection.json', dict(selected=selected, criterion=t.selection,
        rows=candidates, coupled_initial_state='unchanged preceding-load parent',
        universal_optimizer_superiority=False, physical_state_promoted=False))
    q, d, parent_H = map(jnp.asarray, (f['q'], f['d'], f['H']))
    carry = (p['net'], p['state']); counter = loose_counter = 0; loose_first = None
    updates = evaluations = 0
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    adjacency = mesh_adjacency(tri, population_size=m.n_material)
    folder = root/case.name; folder.mkdir()
    started = time.perf_counter()
    def damage(H, old_d, previous): return train(selected, H, previous)
    for sweep in range(1, case.max_sweeps+1):
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
        updates += fields['inner']['iterations']; evaluations += fields['inner']['evaluations']
        count = optimizer_count(carry[1], kind='adam') if selected == 'adam' else None
        if selected == 'adam' and count != a.parent_count+sweep*t.inner_updates:
            raise RuntimeError('full-domain Adam count reset')
        row = dict(case=case.name, solver='pinn', optimizer=selected, omega=1., sweep=sweep,
            load=a.load, **changes, field_stopping=stopping, loose_first_stable=loose_first,
            mechanics_calls=sweep, reaction=fields['reaction'], reaction_damage='d_previous',
            q_d_synchronized=False, inner_AT2=fields['inner']['AT2_relative'],
            stage_AT2=fields['inner']['AT2_relative'], full_loss=fields['inner']['full_loss'],
            optimization=fields['inner'], training_updates=updates, function_evaluations=evaluations,
            full_node_count=m.n_material, mechanics=fields['mechanics'], damage_max=float(d.max()),
            optimizer_count=count, learning_rate=float(schedule(count-1)) if count is not None else None,
            seconds=time.perf_counter()-started)
        saved = dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H), H_used=np.asarray(H),
            damage_image=np.asarray(fields['damage_image']), q_previous=np.asarray(fields['q_previous']),
            d_previous=np.asarray(fields['d_previous']), H_parent=f['H'], d_parent=f['d'])
        atomic_json(folder/f'sweep_{sweep:04d}.json', row)
        if sweep in a.checkpoints or sweep == case.max_sweeps:
            checkpoint(folder/f'outer_{sweep:04d}', saved, dict(row), carry, selected)
        atomic_json(root/'result.json', dict(complete=False, selected=selected, current=row))
        if sweep == 1 or sweep % 8 == 0:
            print(json.dumps(dict(event='full_domain_sweep', **row)), flush=True)
    q_sync, reaction, mechanical = solve(d, a.load, q)
    H_sync = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q_sync))))
    force = m.matrix(np.asarray(d))@np.asarray(q_sync)
    norm = float(np.linalg.norm(force[m.free]))
    if norm > mechanical['convergence_limit']*(1+1e-3): raise RuntimeError('full-domain final host force check fails')
    np.testing.assert_allclose(force@m.load_direction, reaction, rtol=1e-9, atol=1e-6)
    saved.update(q_stage=saved['q'], q=np.asarray(q_sync), H=np.asarray(H_sync),
        raw_current_image=np.asarray(image(carry[0], p['full'](H_sync))))
    physical = residual_metrics(np.asarray(d), np.asarray(H_sync), D, gc0=p['g0'], gc1=p['g1'])[1]
    final = dict(row, reaction=float(reaction), reaction_damage='d', q_d_synchronized=True,
        mechanics=mechanical, mechanics_calls=sweep+1, final_sync_calls=1, host_force_norm=norm,
        physical_AT2=physical, original_residual_pass=bool(physical <= c.reporting.coupled_AT2_relative_limit),
        field_stable=stopping['converged'], certified_path=False, load_advanced=False,
        status='equal_outer_budget_complete', seconds=time.perf_counter()-started,
        crack=crack_extent(np.asarray(d), m.coordinates[:m.n_material], adjacency,
            threshold=.9, tip=[.5, .5], tip_radius=c.physical.l0, right_edge=1., tolerance=1e-12))
    checkpoint(folder/'final', saved, final, carry, selected)
    atomic_json(folder/'result.json', final)
    for name, digest in a.pins.items():
        if sha(Path(a.source_root)/name) != digest: raise RuntimeError('full-domain parent changed')
    for name, digest in code.items():
        if sha(repo/name) != digest: raise RuntimeError('full-domain source changed')
    atomic_json(root/'result.json', dict(complete=True, selected=selected, current=final,
        fixed_load=a.load, load_advanced=False))
    print(json.dumps(dict(event='full_domain_complete', **final)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    run(runpy.run_path(args.config)['get_config'](), args.config, preflight_only=args.preflight_only)
