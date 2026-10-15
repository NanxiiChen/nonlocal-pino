"""Short own-history coupled comparison of certified bounded neural outputs."""
import argparse
import json
from pathlib import Path
import runpy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_optimization import optimize_lbfgs
from examples.fracture2d._legacy.local_damage_trial import at2_metrics, sha
from examples.fracture2d._legacy.outer_stopping import evaluate, validate_controls
from examples.fracture2d._legacy.output_damage_trial import setup, make_problem
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def parent_state(c):
    """Check the accepted prefix ancestry, rather than trusting a directory name."""
    parent = Path(c.head_loading.parent)
    last = json.loads((parent/'result.json').read_text())
    chain = None
    for i in range(1, last['index']+1):
        folder = parent.parent/f'state_{i:04d}'
        r = json.loads((folder/'result.json').read_text())
        if (r['status'] != 'ready' or not r['physical_state_promoted']
                or not r['outer_convergence']['converged']
                or not all(a['pass'] for a in r['outer_convergence']['checks'].values())
                or r['parent_sha256'] != chain
                or r['state_sha256'] != sha(folder/'fields.npz')
                or r['network_sha256'] != sha(folder/'network.eqx')):
            raise ValueError(f'bounded-head coupled parent prefix fails at{i}')
        chain = r['state_sha256']
    if (chain != c.head_loading.parent_field_sha256
            or last['network_sha256'] != c.head_loading.parent_network_sha256
            or last['index']+1 != c.head_loading.indices[0]):
        raise ValueError('bounded-head coupled parent pins/first load differ')
    with np.load(parent/'fields.npz', allow_pickle=False) as f:
        fields = {key: f[key].copy() for key in ('q', 'd', 'H')}
    return fields, last


def run(c):
    validate_controls(c.outer_stop)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or not c.outer_stop.enabled or c.stagger.omega != .5
            or list(c.head_loading.modes) != ['sigmoid', 'exp_square']
            or list(c.head_loading.methods) != ['residual_l32', 'residual_l16']
            or not c.head_loading.indices
            or not 0 < c.head_loading.first_H_relative_limit <= 1e-5
            or any(b != a+1 for a, b in zip(c.head_loading.indices, c.head_loading.indices[1:]))
            or c.head_loading.lbfgs.residual_limit != c.outer_stop.inner_AT2_relative_limit):
        raise ValueError('explicit serialized bounded-head original-gate controls required')
    root = Path(c.head_loading.output_dir)
    root.mkdir()
    started = time.perf_counter()
    fields, parent_report = parent_state(c)
    s = setup(c)
    fixed = Path(c.head_loading.fixed_H_root)
    audit = json.loads((fixed/'audit.json').read_text())
    metadata = json.loads((fixed/'identity.json').read_text())
    if (not audit['pass_replay'] or not audit['source_unchanged']
            or metadata['source'] != s['source_hashes']
            or metadata['geometry'] != s['geometry']):
        raise ValueError('bounded-head fixed-H audit/source geometry differs')
    # Extra loading fields are explicit; the fixed-H numerical controls must match.
    fixed_config = c.to_dict()
    del fixed_config['head_loading']
    if fixed_config != metadata['config']:
        raise ValueError('bounded-head fixed-H configuration differs')
    for name, digest in metadata['code'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('bounded-head certified decoder code differs')
    mechanics = s['mechanics']
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                                  maxiter=c.validation.cg_maxiter)
    matvec, _ = make_padded_matvec(s['operator'])
    matvec = jax.jit(matvec)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    protected = dict(s['source_hashes'])
    for path in (fixed/'identity.json', fixed/'audit.json',
                 Path(c.head_loading.parent)/'fields.npz',
                 Path(c.head_loading.parent)/'result.json',
                 Path(c.head_loading.parent)/'network.eqx'):
        protected[str(path)] = sha(path)
    summaries = []
    for mode in c.head_loading.modes:
        seed = fixed/mode
        report = json.loads((seed/'result.json').read_text())
        if not report['ready']:
            raise ValueError('uncertified fixed-H head cannot enter coupled loading')
        for filename, key in [('network.eqx', 'network_sha256'), ('fields.npz', 'fields_sha256'),
                              ('parameterization.npz', 'parameterization_sha256')]:
            if sha(seed/filename) != report[key]:
                raise ValueError('bounded-head certified seed differs')
            protected[str(seed/filename)] = report[key]
        with np.load(seed/'parameterization.npz', allow_pickle=False) as f:
            columns, gain = f['column_scale'].copy(), float(f['trunk_gain'])
        image, metrics, objectives, _ = make_problem(c, s, mode, columns, gain)
        predict, measure = eqx.filter_jit(image), eqx.filter_jit(metrics)
        net = eqx.tree_deserialise_leaves(seed/'network.eqx', s['net'])
        with np.load(seed/'fields.npz', allow_pickle=False) as f:
            np.testing.assert_allclose(predict(net, s['H'], s['load']), f['damage'], rtol=1e-11, atol=1e-13)
        head_root = root/mode
        head_root.mkdir()
        atomic_json(head_root/'identity.json', dict(schema='own-parent-bounded-neural-loading-v1',
            config=c.to_dict(), mode=mode, geometry=s['geometry'], fixed_H_identity_sha256=sha(fixed/'identity.json'),
            fixed_H_seed=report, parent=parent_report, parent_identity_sha256=sha(Path(c.head_loading.parent).parent/'identity.json'),
            runner_sha256=sha(__file__), own_history=True, direct_targets=False,
            damage_correction=False, inherited_ready_loads=parent_report['index']))
        q_parent, d_parent, H_parent = [jnp.asarray(fields[k]) for k in ('q', 'd', 'H')]
        parent_sha = parent_report['state_sha256']
        curve = []
        for index in c.head_loading.indices:
            load_value, load = index*c.data.load_step, jnp.asarray(index*c.data.load_step)
            folder = head_root/f'state_{index:04d}'
            folder.mkdir()
            begin = time.perf_counter()
            d = d_parent
            q, reaction, mechanical = solve_q(d, load, q_parent)
            H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
            if index == c.head_loading.indices[0]:
                difference = float(np.max(abs(np.asarray(H)-s['H_np'])))
                relative = difference/float(np.max(abs(s['H_np'])))
                atomic_json(folder/'initial_history_replay.json', dict(
                    difference_inf=difference, relative_inf=relative,
                    relative_limit=c.head_loading.first_H_relative_limit,
                    own_FEM_history=True, direct_history_import=False))
                if relative > c.head_loading.first_H_relative_limit:
                    raise ValueError('first own FEM history differs from fixed-H diagnostic beyond declared precision')
            count, rows, status = 0, [], 'stagger_budget_exhausted'
            for sweep in range(1, c.outer_stop.max_sweeps+1):
                previous_q, previous_d, previous_H, previous_R = q, d, H, reaction
                H_used = H
                baseline = np.asarray(s['frozen_image'](H_used, load))
                if not np.isfinite(baseline).all() or baseline.min() < 0 or baseline.max() >= 1:
                    raise ValueError('frozen own-neural bounded initialization is out of range at current H')
                optimizations = []
                for method in c.head_loading.methods:
                    def progress(row):
                        print(json.dumps(dict(event='inner', mode=mode, index=index,
                            sweep=sweep, method=method, **row)), flush=True)
                    net, optimization = optimize_lbfgs(net, objectives[method], metrics, H_used, load,
                        c.head_loading.lbfgs.to_dict(), progress=progress)
                    optimizations.append(dict(method=method, **optimization))
                    if optimization['status'] == 'residual_reached':
                        break
                damage_image = predict(net, H_used, load)
                inner = {k: float(v) for k, v in measure(net, H_used, load).items()}
                atomic_json(folder/f'inner_{sweep:04d}.json', dict(stages=optimizations, metrics=inner))
                if optimization['status'] != 'residual_reached':
                    status = 'inner_rejected'
                    break
                d = d+c.stagger.omega*(damage_image-d)
                q, reaction, mechanical = solve_q(d, load, q)
                H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
                field_scale = max(float(jnp.max(abs(H))), float(jnp.max(abs(previous_H))), c.outer_stop.scale_floor)
                ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:mechanics.n_material], axis=1)/load_value
                row = dict(index=index, load=load_value, sweep=sweep, reaction=reaction,
                    damage_min=float(jnp.min(d)), damage_max=float(jnp.max(d)),
                    fixed_H_image_damage_min=inner['damage_min'], fixed_H_image_damage_max=inner['damage_max'],
                    fixed_H_image_AT2_relative=inner['AT2_relative'],
                    coupled_AT2_relative=at2_metrics(d, H, matvec, gc0, gc1),
                    raw_current_H_AT2_relative=float(measure(net, H, load)['AT2_relative']),
                    damage_change=float(jnp.max(abs(d-previous_d))),
                    q_change_over_load=float(jnp.max(abs(q-previous_q)))/load_value,
                    H_change_relative=float(jnp.max(abs(H-previous_H)))/field_scale,
                    reaction_change_relative=abs(reaction-previous_R)/max(abs(reaction), abs(previous_R), c.outer_stop.scale_floor),
                    healing=float(jnp.max(jnp.maximum(d_parent-d, 0.))),
                    history_decrease=float(jnp.max(jnp.maximum(H_parent-H, 0.))),
                    drift_count=int(np.count_nonzero(ratio > direct.DRIFT_ENVELOPE)),
                    mechanics=mechanical, parent_sha256=parent_sha)
                outer = evaluate(row, c.outer_stop, c.reporting, sweep=sweep, previous_consecutive=count)
                raw_pass = row['raw_current_H_AT2_relative'] <= c.reporting.coupled_AT2_relative_limit
                outer['checks']['raw_current_H_AT2'] = dict(value=row['raw_current_H_AT2_relative'],
                    limit=c.reporting.coupled_AT2_relative_limit, **{'pass': raw_pass})
                if not raw_pass:
                    outer.update(pass_this_sweep=False, consecutive=0, converged=False, reason='raw_neural_current_H_rejected')
                count = outer['consecutive']
                row['outer_convergence'] = outer
                rows.append(row)
                atomic_json(folder/f'sweep_{sweep:04d}.json', row)
                print(json.dumps(dict(event='outer', mode=mode, index=index, sweep=sweep,
                    current_AT2=row['coupled_AT2_relative'], raw_current_AT2=row['raw_current_H_AT2_relative'],
                    damage_change=row['damage_change'], ready=outer['converged'])), flush=True)
                if outer['converged']:
                    status = 'ready'
                    break
            np.savez_compressed(folder/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
                H_used=np.asarray(H_used), damage_image=np.asarray(damage_image),
                d_parent=np.asarray(d_parent), H_parent=np.asarray(H_parent),
                raw_current_image=np.asarray(predict(net, H, load)))
            eqx.tree_serialise_leaves(folder/'network.eqx', net)
            force = mechanics.matrix(np.asarray(d))@np.asarray(q)
            host_force = float(np.linalg.norm(force[mechanics.free]))
            if host_force > mechanical['convergence_limit']*(1+1e-3):
                raise RuntimeError('bounded neural FEM force replay failed')
            record = dict(index=index, load=load_value, status=status, parent_sha256=parent_sha,
                state_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
                inner=inner, outer_convergence=rows[-1]['outer_convergence'] if rows else None,
                inner_calls=sweep, accepted_inner_sweeps=len(rows), reaction=reaction,
                seconds=time.perf_counter()-begin, host_force_norm=host_force,
                host_reaction=float(force@mechanics.load_direction), physical_state_promoted=status=='ready')
            atomic_json(folder/'result.json', record)
            curve.append(record)
            atomic_json(head_root/'result.json', dict(complete=True, status=status, curve=curve,
                inherited_ready_loads=parent_report['index'], new_ready_loads=sum(a['status']=='ready' for a in curve)))
            if status != 'ready':
                break
            q_parent, d_parent, H_parent, parent_sha = q, d, H, record['state_sha256']
        summaries.append(dict(mode=mode, curve=curve))
        atomic_json(root/'result.json', dict(complete=len(summaries)==len(c.head_loading.modes),
            reports=summaries, seconds=time.perf_counter()-started))
    if protected != {path: sha(path) for path in protected}:
        raise RuntimeError('bounded-head loading parent/seed changed')
    return summaries


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    c = runpy.run_path(p.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
