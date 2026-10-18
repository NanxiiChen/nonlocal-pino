"""Local-mechanics stagger with nonlocal or explicit experimental P1 damage."""
import json
import os
from pathlib import Path
import shutil
import time
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.at2_cg import damage_solver as direct_damage_solver
from examples.fracture2d.losses import damage_weights, residual_mean, residual_metrics
from examples.fracture2d.mesh import load_geometry, prepare
from examples.fracture2d.stagger import committed_history, damage_change_limit, field_changes, field_stop, mechanics_first_sweep
from examples.fracture2d.storage import atomic_fields, atomic_json, load_fields


def validate(c):
    a = c.loading
    material = [c.physical.E, c.physical.nu, c.physical.Gc, c.physical.l0, c.physical.k_res]
    if (not np.isfinite(material).all() or min(c.physical.E, c.physical.Gc, c.physical.l0) <= 0
            or not 0 <= c.physical.nu < .5 or not 0 < c.physical.k_res < 1
            or type(c.train.seed) is not int or c.train.seed < 0
            or type(c.sample.seed) is not int or c.sample.seed < 0):
        raise ValueError('invalid explicit physical material or initialization/sampling seeds')
    if (not np.isfinite([a.du, a.u_final, a.critical_load, a.omega]).all()
            or not 0 < a.du <= a.u_final or not 0 < a.critical_load
            or not 0 < a.omega <= 1):
        raise ValueError('invalid explicit loading increments/terminal/relaxation')
    n = round(a.u_final/a.du)
    if not np.isclose(n*a.du, a.u_final, rtol=0, atol=1e-14):
        raise ValueError('terminal load must be an integer multiple of du')
    if (c.arch.type not in ('mlp', 'history_operator', 'conditioned_history_operator')
            or c.loss.norm not in ('l1', 'l2', 'l16', 'energy') or not 0 < c.loss.floor < 1
            or c.damage.discretization not in ('nonlocal', 'fem')
            or c.sample.mode not in ('full', 'random')
            or c.train.optimizer not in ('adam', 'soap', 'lbfgs')
            or c.arch.output_head not in ('sigmoid', 'exp_square', 'identity')
            or c.runtime.platform not in ('cpu', 'gpu')
            or c.loss.weight_mode not in ('uniform', 'damage', 'reciprocal')
            or not np.isfinite([c.loss.weight_boost, c.loss.weight_power,
                c.loss.weight_epsilon, c.loss.weight_cap]).all()
            or c.loss.weight_boost < 0 or min(c.loss.weight_power,
                c.loss.weight_epsilon, c.loss.weight_cap) <= 0
            or not 0 <= c.validation.range_atol <= 1e-3):
        raise ValueError('invalid explicit loss/sampling/optimizer/head/runtime setting')
    if c.damage.discretization == 'fem' and (
            c.arch.type != 'conditioned_history_operator' or c.loss.norm != 'energy'
            or c.sample.mode != 'full' or c.loss.weight_mode != 'uniform'):
        raise ValueError('FEM damage requires conditioned history, full uniform energy training')
    if c.loss.norm == 'energy' and c.damage.discretization != 'fem':
        raise ValueError('energy training currently requires explicit FEM damage')
    integers = (a.prefix_outer_steps, a.outer_steps, a.minimum_sweeps,
        a.consecutive_passes, c.train.inner_steps, c.sample.core_count,
        c.sample.pool_size, c.sample.resample_every,
        c.output.figure_every, c.output.figure_dpi, c.output.scalar_every, c.output.cpu_threads)
    if any(type(v) is not int or v < 1 for v in integers):
        raise ValueError('iteration/sampling/monitor capacities must be positive integers')
    if (a.minimum_sweeps > min(a.prefix_outer_steps, a.outer_steps)
            or min(a.damage_tolerance, a.displacement_tolerance) <= 0
            or not np.isfinite([a.damage_tolerance, a.displacement_tolerance]).all()
            or any(type(v) is not int or v < 1 for v in c.output.checkpoint_sweeps)):
        raise ValueError('invalid explicit field-stopping/checkpoint controls')
    if c.train.optimizer == 'lbfgs' and c.sample.mode != 'full':
        raise ValueError('L-BFGS needs a fixed full objective; choose Adam/SOAP for random sampling')
    if any(type(v) is not bool for v in
           (a.field_stop, a.advance_on_budget, c.output.resume, c.output.tensorboard, c.mesh.prepare)):
        raise ValueError('boolean policies must be explicit booleans')
    if 'damage_stop' in a:
        damage_change_limit(a, damage_max=0.)
        if not a.field_stop or a.advance_on_budget or c.arch.output_head != 'sigmoid':
            raise ValueError('scaled damage stopping requires sigmoid, field stopping and no budget advancement')
    if 'rate_control' in c.train:
        from examples.fracture2d.learning_rate import BudgetLearningRate
        control = BudgetLearningRate(c.train.rate_control,
            initial_rate=c.train.learning_rate, minimum_rate=c.train.minimum_rate,
            decay_rate=c.train.decay_rate)
        if (c.damage.discretization != 'fem' or c.arch.type != 'conditioned_history_operator'
                or c.train.optimizer != 'adam' or c.history.replay_enabled
                or not a.field_stop or a.advance_on_budget
                or a.prefix_outer_steps != control.outers_per_rate
                or a.outer_steps != control.outers_per_rate):
            raise ValueError('budget rate control requires replay-off FEM Adam, field stopping, '
                             'no budget advancement, and matching outer blocks')
    return np.arange(1, n+1, dtype=np.float64)*a.du


def canonical_config(c):
    values = c.to_dict()
    del values['output']['resume']
    return values


def save_checkpoint(folder, fields, row, pinn):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_fields(folder/'fields.npz', **fields)
    row['fields_path'] = str(folder/'fields.npz')
    if pinn is not None:
        pinn.save(folder)
        row['network_path'] = str(folder/'network.eqx')
        if pinn.state is not None:
            row['optimizer_path'] = str(folder/'optimizer.eqx')
    atomic_json(folder/'result.json', row)


def run(c, *, damage_solver):
    if damage_solver not in ('direct', 'pinn'):
        raise ValueError('unknown damage solver')
    loads = validate(c)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:c.output.cpu_threads])
    if jax.devices()[0].platform != c.runtime.platform:
        raise ValueError(f'required {c.runtime.platform} backend is unavailable')
    if c.mesh.prepare:
        prepare(c)
        return Path(c.mesh.prepared_dir)
    m, operator, geometry = load_geometry(c)
    if c.damage.discretization == 'fem' and c.output.reference_root:
        reference = json.loads((Path(c.output.reference_root)/'identity.json').read_text())
        if (reference['schema'] != 'local-p1-lumped-at2-v1'
                or reference['damage_solver'] != 'direct' or reference['geometry'] != geometry
                or reference['config']['physical'] != c.physical.to_dict()):
            raise ValueError('FEM monitoring requires a matched P1 direct reference: schema, geometry and material must agree')
    parity = json.loads((Path(c.mesh.prepared_dir)/'fenics_parity.json').read_text())
    if not parity['pass']:
        raise ValueError('local mechanics geometry has not passed independent FEM parity')
    root = Path(c.output.root)
    if root.exists() and not c.output.resume:
        raise ValueError(f'refusing to overwrite {root}; choose a fresh output root')
    if c.output.resume and not (root/'identity.json').is_file():
        raise ValueError('resume requires an existing run identity')
    root.mkdir(parents=True, exist_ok=True)
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    D = operator.to_scipy()
    solve_q = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                           maxiter=c.validation.cg_maxiter)
    pinn, solve_d = None, None
    history_pinn = damage_solver == 'pinn' and c.arch.type in ('history_operator', 'conditioned_history_operator')
    rate_controlled = 'rate_control' in c.train
    if rate_controlled and not history_pinn:
        raise ValueError('budget rate control is only defined for history-network damage training')
    if history_pinn:
        from examples.fracture2d.history_training import HistoryDamagePINN
        pinn = HistoryDamagePINN(c, m, operator)
    elif damage_solver == 'pinn':
        from examples.fracture2d.train import DamagePINN
        pinn = DamagePINN(c, m, operator)
    else:
        solve_d = direct_damage_solver(operator, m.volumes, g0, g1,
            tol=c.validation.cg_rtol, atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)
        if c.loss.norm == 'energy':
            from examples.fracture2d.fem_damage import make_energy
            direct_energy = jax.jit(make_energy(operator, m.volumes,
                gc0=g0, gc1=g1, floor=c.loss.floor))
    sources = [str(f) for f in Path(__file__).parent.glob('*.py')]
    inputs = [str(Path(c.mesh.prepared_dir)/name)
              for name in ('identity.json', 'mesh.npz', 'damage_operator.npz', 'fenics_parity.json')]
    if c.damage.discretization == 'fem':
        inputs.remove(str(Path(c.mesh.prepared_dir)/'damage_operator.npz'))
    if pinn is not None and not history_pinn:
        inputs.append(c.data.basis_path)
    identity = dict(schema=('local-p1-lumped-at2-v1' if c.damage.discretization == 'fem'
                            else 'local-p1-nonlocal-at2-v1'), config=canonical_config(c),
        damage_solver=damage_solver, geometry=geometry, sources=sources, inputs=inputs,
        history_floor='preceding_load_endpoint', reference_training_targets=False,
        residual_is_stopping_gate=False, physical_initial_damage=0., physical_initial_history=0.)
    if c.output.resume:
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('resume configuration, solver or file paths differ')
    else:
        atomic_json(root/'identity.json', identity)
        for path in sources:
            snapshot = root/'code_snapshot'/Path(path).name
            snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, snapshot)
        atomic_json(root/'config.json', c.to_dict())
    if pinn is not None and pinn.pool_info is not None:
        atomic_json(root/'sampling_pool.json', pinn.pool_info)
        print(json.dumps(dict(event='sampling_pool', **pinn.pool_info)), flush=True)
    q = jnp.zeros(len(m.load_direction), jnp.float64)
    d = jnp.zeros(m.n_material, jnp.float64)
    H = jnp.zeros_like(d)
    if not c.output.resume:
        zero = dict(index=0, load=0., reaction=0., optimizer_count=0,
                    q_d_synchronized=True, status='exact_physical_zero')
        save_checkpoint(root/'state_0000', dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H)), zero, pinn)
        if history_pinn:
            atomic_json(root/'history_parameterization.json', dict(
                network='point-patch-coordinate-attention',
                tokens=c.history.coarse_cells**2+c.history.fine_tokens,
                bank_shape=list(pinn.replay.histories.shape), initial_replay_count=int(pinn.replay.count),
                history_inputs='own mechanics, first outer and synchronized endpoint'))
        elif pinn is not None:
            atomic_fields(root/'parameterization.npz', column_scale=pinn.scales,
                          trunk_gain=np.float64(c.arch.trunk_gain))
    if c.output.resume:
        zero = root/'state_0000'
        zero_report = json.loads((zero/'result.json').read_text())
        if zero_report['status'] != 'exact_physical_zero':
            raise ValueError('saved initial state status differs')
        with np.load(zero/'fields.npz', allow_pickle=False) as fields:
            for name, initial in [('q', q), ('d', d), ('H', H)]:
                if not np.array_equal(fields[name], np.asarray(initial)):
                    raise ValueError('saved initial physical state is not exact zero')
    curve, monitor = [], None
    if c.output.tensorboard:
        from examples.fracture2d.monitoring import Monitor
        monitor = Monitor(root, c, m, damage_solver.upper())
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    monitor_span = c.loading.outer_steps
    if rate_controlled:
        control = pinn.rate_control
        monitor_span *= (control.max_blocks_per_load if control.mode == 'progress_blocks' else len(control.rates))
    started = time.perf_counter()
    try:
        for index, target in enumerate(loads, 1):
            folder = root/f'state_{index:04d}'
            if c.output.resume and (folder/'result.json').is_file():
                report = json.loads((folder/'result.json').read_text())
                if report['status'] == 'budget_exhausted' and not c.loading.advance_on_budget:
                    raise ValueError('cannot resume past a budget-exhausted load; refine it explicitly first')
                if (report['q_d_synchronized'] is not True or report['load'] != float(target)
                        or report['parent_path'] != str(root/f'state_{index-1:04d}'/'fields.npz')):
                    raise ValueError('saved endpoint load or parent path differs')
                fields = load_fields(folder/'fields.npz', nodes=m.n_material, dofs=len(m.load_direction))
                m.factors(fields['d'])
                if np.any(fields['H'] < np.asarray(H)):
                    raise ValueError('saved load history decreases relative to its parent')
                q, d, H = [jnp.asarray(fields[k]) for k in ('q', 'd', 'H')]
                if pinn is not None:
                    pinn.restore(folder, report['optimizer_count'])
                curve.append(report)
                continue
            load = float(target)
            parent_H, parent_d = H, d
            cap = c.loading.outer_steps if load >= c.loading.critical_load-1e-14 else c.loading.prefix_outer_steps
            consecutive, loss_value = 0, 0.
            block_reaction, control_calls = None, 0
            if history_pinn:
                pinn.begin_load(index, load)

            def damage(history, old, carry):
                nonlocal loss_value
                image = solve_d(history, old) if pinn is None else None
                if pinn is not None:
                    image, loss_value = pinn.solve(history, old, load)
                m.factors(np.asarray(image))
                raw, relative = residual_metrics(np.asarray(image), np.asarray(history), D, gc0=g0, gc1=g1)
                if pinn is None:
                    if c.loss.norm == 'energy':
                        loss_value = float(direct_energy(image, history))
                    else:
                        loss_value = float(residual_mean(jnp.asarray(raw), history,
                            jnp.full(m.n_material, 1/m.n_material), damage_weights(old, c.loss),
                            population_size=m.n_material, norm=c.loss.norm, floor=c.loss.floor))
                return image, carry, dict(AT2_relative=relative)

            sweep = 0
            while sweep < cap:
                sweep += 1
                stage, _ = mechanics_first_sweep(q, d, parent_H, load, None,
                    mechanics=solve_q, driver=m.driver, damage=damage, omega=c.loading.omega)
                q, d, H = (stage[k] for k in ('q', 'd', 'H'))
                if block_reaction is None:
                    block_reaction = stage['reaction']
                changes = field_changes(q, stage['q_previous'], d, stage['d_previous'], tri, area, load=load)
                damage_max = float(d.max())
                damage_limit = damage_change_limit(c.loading, damage_max=damage_max)
                stop = field_stop(changes, displacement_limit=c.loading.displacement_tolerance,
                    damage_limit=damage_limit, sweep=sweep,
                    minimum_sweeps=c.loading.minimum_sweeps,
                    consecutive_passes=c.loading.consecutive_passes, previous_consecutive=consecutive)
                consecutive = stop['consecutive']
                row = dict(index=index, load=load, sweep=sweep, outer_cap=cap, **changes,
                    field_stable=stop['converged'], field_stopping=stop, full_loss=loss_value,
                    inner_AT2=stage['inner']['AT2_relative'], reaction=stage['reaction'],
                    damage_min=float(d.min()), damage_max=damage_max, damage_tolerance=damage_limit,
                    optimizer_count=0 if pinn is None else pinn.count,
                    q_d_synchronized=False, mechanics_calls=sweep+control_calls, mechanics=stage['mechanics'],
                    parent_path=str(root/f'state_{index-1:04d}'/'fields.npz'))
                if history_pinn:
                    row.update(pinn.diagnostics())
                if sweep in c.output.checkpoint_sweeps:
                    save_checkpoint(folder/f'outer_{sweep:04d}', dict(q=np.asarray(q),
                        d=np.asarray(d), H=np.asarray(H), H_parent=np.asarray(parent_H)), dict(row), pinn)
                if sweep == 1 or sweep % c.output.scalar_every == 0:
                    atomic_json(root/'result.json', dict(complete=False, curve=curve, current=row))
                    print(json.dumps(dict(event='outer', **row)), flush=True)
                    if monitor is not None:
                        monitor.stage(row, (index-1)*monitor_span+sweep)
                if c.loading.field_stop and stop['converged']:
                    break
                if sweep == cap and rate_controlled:
                    control = pinn.rate_control
                    if control.mode == 'progress_blocks':
                        # Diagnostic solve only: preserve raw q/H and preceding-load floor.
                        q_control, reaction_control, mechanical_control = solve_q(d, load, q)
                        control_calls += 1
                        force_control = m.matrix(np.asarray(d))@np.asarray(q_control)
                        if np.linalg.norm(force_control[m.free]) > mechanical_control['convergence_limit']*(1+1e-3):
                            raise RuntimeError('progress-block independent host force check failed')
                        np.testing.assert_allclose(force_control@m.load_direction, reaction_control, rtol=1e-9, atol=1e-6)
                        decision = control.progress_action(block=sweep//control.outers_per_rate,
                            previous_reaction=block_reaction, reaction=reaction_control)
                        if decision['action'] in ('continue', 'lower'):
                            cap += control.outers_per_rate
                        print(json.dumps(dict(event='progress_block', index=index, load=load,
                            after_sweep=sweep, previous_reaction=block_reaction,
                            synchronized_reaction=reaction_control, outer_cap=cap,
                            **decision, **pinn.diagnostics())), flush=True)
                        block_reaction = reaction_control
                    elif control.lower():
                        previous_cap = cap
                        cap += control.outers_per_rate
                        print(json.dumps(dict(event='learning_rate_lowered', index=index,
                            load=load, after_sweep=sweep, previous_cap=previous_cap,
                            outer_cap=cap, **pinn.diagnostics())), flush=True)
            # Save mechanically synchronized endpoints; keep preceding-load H floor.
            q_sync, reaction, mechanical = solve_q(d, load, q)
            H_sync = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q_sync))))
            force = m.matrix(np.asarray(d)) @ np.asarray(q_sync)
            if np.linalg.norm(force[m.free]) > mechanical['convergence_limit']*(1+1e-3):
                raise RuntimeError('endpoint independent host FEM force check failed')
            np.testing.assert_allclose(force @ m.load_direction, reaction, rtol=1e-9, atol=1e-6)
            _, physical = residual_metrics(np.asarray(d), np.asarray(H_sync), D, gc0=g0, gc1=g1)
            report = dict(row, reaction=reaction, physical_AT2=physical, mechanics=mechanical,
                q_d_synchronized=True, mechanics_calls=sweep+control_calls+1, certified_path=False,
                status='field_stable' if stop['converged'] else 'budget_exhausted')
            if history_pinn:
                pinn.end_load(H_sync)
                report.update(pinn.diagnostics())
            fields = dict(q=np.asarray(q_sync), d=np.asarray(d), H=np.asarray(H_sync),
                H_used=np.asarray(H), H_parent=np.asarray(parent_H), d_parent=np.asarray(parent_d))
            save_checkpoint(folder, fields, report, pinn)
            curve.append(report)
            q, H = q_sync, H_sync
            atomic_json(root/'result.json', dict(complete=False, curve=curve, current=report))
            print(json.dumps(dict(event='load_complete', **report)), flush=True)
            if monitor is not None:
                monitor.endpoint(report, fields, curve)
            if not stop['converged'] and not c.loading.advance_on_budget:
                raise RuntimeError(f'load {load} exhausted outer budget; endpoint saved, advancement stopped')
        atomic_json(root/'result.json', dict(complete=True, curve=curve, current=curve[-1],
            certified_path=False, seconds=time.perf_counter()-started))
        return root/'result.json'
    finally:
        if monitor is not None:
            monitor.close()
