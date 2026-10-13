"""Certified sigmoid continuation with own history and accepted-only restart."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import at2_metrics, sha
from examples.fracture2d.outer_stopping import evaluate, validate_controls
from examples.fracture2d.output_damage_loading import parent_state
from examples.fracture2d.output_damage_trial import setup, make_problem
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def accepted_state(folder, expected_parent):
    report = json.loads((folder/'result.json').read_text())
    if (report['status'] != 'ready' or not report['physical_state_promoted']
            or report['parent_sha256'] != expected_parent
            or report['state_sha256'] != sha(folder/'fields.npz')
            or report['network_sha256'] != sha(folder/'network.eqx')
            or not report['outer_convergence']['converged']
            or not all(a['pass'] for a in report['outer_convergence']['checks'].values())):
        raise ValueError(f'cannot advance rejected/changed/unready continuation state: {folder}')
    with np.load(folder/'fields.npz', allow_pickle=False) as f:
        fields = {k: f[k].copy() for k in ('q', 'd', 'H')}
    return fields, report


def verify_parent(c):
    """Validate the old zero-origin prefix followed by certified new33/34."""
    previous, report = parent_state(c)
    protected = {}
    legacy = Path(c.head_loading.parent).parent
    for i in range(1, report['index']+1):
        for name in ('fields.npz', 'network.eqx', 'result.json'):
            path = legacy/f'state_{i:04d}'/name
            protected[str(path)] = sha(path)
    parent = Path(c.continuation.parent)
    identity = json.loads((parent.parent/'identity.json').read_text())
    base = c.to_dict()
    del base['continuation']
    if (identity['config'] != base or identity['mode'] != c.continuation.mode
            or identity['runner_sha256'] != sha(Path(__file__).with_name('output_damage_loading.py'))
            or not identity['own_history'] or identity['direct_targets'] or identity['damage_correction']):
        raise ValueError('certified bounded parent model/configuration/code differs')
    audit_path = parent.parent.parent/'audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['pass_replay'] or not audit['own_history'] or not audit['parent_unchanged']:
        raise ValueError('bounded parent lacks independent own-history audit')
    for path in (parent.parent/'identity.json', audit_path, legacy/'identity.json'):
        protected[str(path)] = sha(path)
    for index in c.head_loading.indices:
        folder = parent.parent/f'state_{index:04d}'
        fields, current = accepted_state(folder, report['state_sha256'])
        with np.load(folder/'fields.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(f['d_parent'], previous['d'])
            np.testing.assert_array_equal(f['H_parent'], previous['H'])
        if (current['index'] != index or current['load'] != index*c.data.load_step
                or np.any(fields['H'] < previous['H'])):
            raise ValueError('bounded parent load/history ancestry differs')
        for name in ('fields.npz', 'network.eqx', 'result.json'):
            path = folder/name
            protected[str(path)] = sha(path)
        previous, report = fields, current
    if (parent.name != f"state_{report['index']:04d}"
            or report['index']+1 != c.continuation.first_index
            or report['state_sha256'] != c.continuation.parent_field_sha256
            or report['network_sha256'] != c.continuation.parent_network_sha256):
        raise ValueError('bounded continuation parent pins/index differ')
    return previous, report, protected


def atomic_fields(path, **fields):
    temporary = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)


def save_pending(folder, net, q, d, H, *, index, load, completed_sweep,
                 consecutive, reaction, mechanical, parent_sha, last_image):
    """Checkpoint only completed outer work; interrupted inner solve restarts."""
    fields_path = folder/f'pending_{completed_sweep:04d}_fields.npz'
    network_path = folder/f'pending_{completed_sweep:04d}_network.eqx'
    fields = dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H))
    if last_image is not None:
        fields.update({k: np.asarray(v) for k, v in last_image.items()})
    atomic_fields(fields_path, **fields)
    temporary = network_path.with_suffix('.tmp.eqx')
    eqx.tree_serialise_leaves(temporary, net)
    temporary.replace(network_path)
    atomic_json(folder/'pending.json', dict(index=index, load=load,
        completed_sweep=completed_sweep, consecutive=consecutive, reaction=reaction,
        mechanical=mechanical, parent_sha256=parent_sha,
        fields_file=fields_path.name, network_file=network_path.name, has_image=last_image is not None,
        fields_sha256=sha(fields_path), network_sha256=sha(network_path)))
    # Keep two complete generations; the manifest changes atomically before
    # deleting older worker-owned restart scratch. Final/sweep files stay.
    for old in folder.glob('pending_*_fields.npz'):
        generation = int(old.name.split('_')[1])
        if generation < completed_sweep-1:
            old.unlink()
            (folder/f'pending_{generation:04d}_network.eqx').unlink()


def load_pending(folder, net, *, index, load, parent_sha):
    meta = json.loads((folder/'pending.json').read_text())
    if (meta['index'] != index or meta['load'] != load or meta['parent_sha256'] != parent_sha
            or Path(meta['fields_file']).name != meta['fields_file']
            or Path(meta['network_file']).name != meta['network_file']
            or meta['fields_sha256'] != sha(folder/meta['fields_file'])
            or meta['network_sha256'] != sha(folder/meta['network_file'])):
        raise ValueError('pending bounded continuation checkpoint differs')
    with np.load(folder/meta['fields_file'], allow_pickle=False) as f:
        q, d, H = [jnp.asarray(f[k]) for k in ('q', 'd', 'H')]
        last_image = {k: jnp.asarray(f[k]) for k in ('H_used', 'damage_image')} if meta['has_image'] else None
    net = eqx.tree_deserialise_leaves(folder/meta['network_file'], net)
    return net, q, d, H, meta, last_image


def run(c, *, resume, parent_loader, setup_fn, extra_code_names):
    validate_controls(c.outer_stop)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or c.continuation.mode != 'sigmoid' or c.stagger.omega != .5
            or not c.outer_stop.enabled
            or not c.continuation.first_index <= c.continuation.last_index <= len(c.data.reference_indices)
            or c.data.reference_indices != list(range(1, 76))
            or c.continuation.lbfgs.to_dict() != c.head_loading.lbfgs.to_dict()
            or list(c.continuation.methods) != list(c.head_loading.methods)):
        raise ValueError('bounded continuation requires explicit original sigmoid schedule/gates/optimizer controls')
    parent_fields, parent_report, protected = parent_loader(c)
    s = setup_fn(c)
    protected.update(s['source_hashes'])
    fixed = Path(c.head_loading.fixed_H_root)
    metadata = json.loads((fixed/'identity.json').read_text())
    configuration = c.to_dict()
    del configuration['continuation']
    del configuration['head_loading']
    if metadata['config'] != configuration or metadata['source'] != s['source_hashes']:
        raise ValueError('certified fixed-H decoder differs')
    for name, digest in metadata['code'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('certified decoder source code changed')
    seed = fixed/c.continuation.mode
    seed_report = json.loads((seed/'result.json').read_text())
    for name, key in [('fields.npz', 'fields_sha256'), ('network.eqx', 'network_sha256'),
                      ('parameterization.npz', 'parameterization_sha256')]:
        if sha(seed/name) != seed_report[key]:
            raise ValueError('certified decoder parameterization changed')
        protected[str(seed/name)] = seed_report[key]
    for path in (fixed/'identity.json', fixed/'audit.json', seed/'result.json'):
        protected[str(path)] = sha(path)
    with np.load(seed/'parameterization.npz', allow_pickle=False) as f:
        columns, gain = f['column_scale'].copy(), float(f['trunk_gain'])
    image, metrics, objectives, _ = make_problem(c, s, c.continuation.mode, columns, gain)
    predict, measure = eqx.filter_jit(image), eqx.filter_jit(metrics)
    parent = Path(c.continuation.parent)
    net = eqx.tree_deserialise_leaves(parent/'network.eqx', s['net'])
    parent_load = jnp.asarray(parent_report['load'])
    with np.load(parent/'fields.npz', allow_pickle=False) as f:
        np.testing.assert_allclose(predict(net, jnp.asarray(f['H_used']), parent_load),
                                   f['damage_image'], rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(predict(net, jnp.asarray(f['H']), parent_load),
                                   f['raw_current_image'], rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(f['H'], np.maximum(f['H_parent'], s['mechanics'].driver(f['q'])))
    fitters, parity = {}, []
    for method in c.continuation.methods:
        fit, vg, ev = make_cached_lbfgs(net, objectives[method], metrics)
        fitters[method] = fit
        arrays, _ = eqx.partition(net, eqx.is_inexact_array)
        flat, _ = ravel_pytree(arrays)
        H = jnp.asarray(parent_fields['H'])
        value, gradient = vg(flat, H, parent_load)
        reference_value, reference_gradient = eqx.filter_jit(eqx.filter_value_and_grad(objectives[method]))(net, H, parent_load)
        reference_flat, _ = ravel_pytree(eqx.filter(reference_gradient, eqx.is_inexact_array))
        difference = float(jnp.max(abs(gradient-reference_flat)))
        scale = max(float(jnp.max(abs(reference_flat))), c.outer_stop.scale_floor)
        if (abs(float(value-reference_value)) > c.validation.gradient_rtol*max(abs(float(reference_value)), c.outer_stop.scale_floor)
                or difference/scale > c.validation.gradient_rtol):
            raise RuntimeError('cached bounded damage objective/gradient parity failed')
        a, b = ev(flat, H, parent_load), measure(net, H, parent_load)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12)
        parity.append(dict(method=method, value=float(value), gradient_difference_inf=difference,
                           gradient_relative_difference=difference/scale))
    root = Path(c.continuation.output_dir)
    canonical = c.to_dict()
    canonical['continuation']['last_index'] = 0
    code_names = ['output_damage_continue.py', 'cached_damage_optimization.py',
        'output_damage_loading.py', 'output_damage_trial.py', 'bounded_damage_output.py',
        'conditioned_damage.py', 'compact_damage_features.py', 'compact_damage_loading.py',
        'damage_optimization.py', 'local_damage_trial.py', 'local_mechanics.py',
        'split_pinn_gradients.py', 'train.py', *extra_code_names]
    code = {name: sha(Path(__file__).with_name(name)) for name in code_names}
    identity = dict(schema='certified-sigmoid-own-history-continuation-v1', config=canonical,
        parent=parent_report, geometry=s['geometry'], inherited_ready_loads=parent_report['index'],
        source=protected, code=code, complete_decoder=True, own_history=True,
        direct_targets=False, damage_correction=False, cached_compilation=True)
    if resume:
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('bounded continuation resume identity differs')
    else:
        root.mkdir()
        atomic_json(root/'identity.json', identity)
        atomic_json(root/'compilation_parity.json', dict(pass_parity=True, reports=parity))
        for name in code_names:
            (root/'code_snapshot').mkdir(exist_ok=True)
            shutil.copyfile(Path(__file__).with_name(name), root/'code_snapshot'/name)
    mechanics = s['mechanics']
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                                  maxiter=c.validation.cg_maxiter)
    matvec, _ = make_padded_matvec(s['operator'])
    matvec = jax.jit(matvec)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    q_parent, d_parent, H_parent = [jnp.asarray(parent_fields[k]) for k in ('q', 'd', 'H')]
    parent_sha = parent_report['state_sha256']
    curve = []
    started = time.perf_counter()
    for index in range(c.continuation.first_index, c.continuation.last_index+1):
        load_value, load = index*c.data.load_step, jnp.asarray(index*c.data.load_step)
        folder = root/f'state_{index:04d}'
        if resume and (folder/'result.json').exists():
            fields, report = accepted_state(folder, parent_sha)
            if report['index'] != index or report['load'] != load_value:
                raise ValueError('continuation accepted load mapping differs')
            with np.load(folder/'fields.npz', allow_pickle=False) as f:
                np.testing.assert_array_equal(f['d_parent'], d_parent)
                np.testing.assert_array_equal(f['H_parent'], H_parent)
                np.testing.assert_array_equal(f['H'], np.maximum(np.asarray(H_parent), mechanics.driver(f['q'])))
            q_parent, d_parent, H_parent = [jnp.asarray(fields[k]) for k in ('q', 'd', 'H')]
            net = eqx.tree_deserialise_leaves(folder/'network.eqx', net)
            parent_sha = report['state_sha256']
            curve.append(report)
            continue
        begin = time.perf_counter()
        if resume and folder.exists():
            net, q, d, H, pending, last_image = load_pending(folder, net, index=index, load=load_value, parent_sha=parent_sha)
            first_sweep, count = pending['completed_sweep']+1, pending['consecutive']
            reaction, mechanical = pending['reaction'], pending['mechanical']
            if not 0 <= pending['completed_sweep'] <= c.outer_stop.max_sweeps or not 0 <= count <= c.outer_stop.max_sweeps:
                raise ValueError('pending outer counter/budget differs')
        else:
            folder.mkdir()
            d = d_parent
            q, reaction, mechanical = solve_q(d, load, q_parent)
            H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
            first_sweep, count = 1, 0
            save_pending(folder, net, q, d, H, index=index, load=load_value, completed_sweep=0,
                consecutive=0, reaction=reaction, mechanical=mechanical, parent_sha=parent_sha, last_image=None)
            last_image = None
        status, last_outer = 'stagger_budget_exhausted', None
        sweep = first_sweep-1
        if last_image is not None:
            H_used, damage_image = last_image['H_used'], last_image['damage_image']
            inner = {k: float(v) for k, v in measure(net, H_used, load).items()}
            last_row = json.loads((folder/f'sweep_{sweep:04d}.json').read_text())
            last_outer = last_row['outer_convergence']
            if last_outer['consecutive'] != count:
                raise ValueError('pending outer confirmation count differs')
            if last_outer['converged']:
                status = 'ready'
                first_sweep = c.outer_stop.max_sweeps+1
        for sweep in range(first_sweep, c.outer_stop.max_sweeps+1):
            previous_q, previous_d, previous_H, previous_R = q, d, H, reaction
            H_used = H
            baseline = np.asarray(s['frozen_image'](H_used, load))
            if not np.isfinite(baseline).all() or baseline.min() < 0 or baseline.max() >= 1:
                atomic_json(folder/'initializer_error.json', dict(index=index, sweep=sweep,
                    minimum=float(baseline.min()), maximum=float(baseline.max()), physical_state_promoted=False))
                raise ValueError('bounded continuation frozen initializer outside certified range')
            optimizations = []
            for method in c.continuation.methods:
                def progress(row):
                    print(json.dumps(dict(event='inner', index=index, sweep=sweep, method=method, **row)), flush=True)
                net, optimization = fitters[method](net, H_used, load,
                    c.continuation.lbfgs.to_dict(), progress=progress)
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
            scale = max(float(jnp.max(abs(H))), float(jnp.max(abs(previous_H))), c.outer_stop.scale_floor)
            ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:mechanics.n_material], axis=1)/load_value
            row = dict(index=index, load=load_value, sweep=sweep, reaction=reaction,
                damage_min=float(jnp.min(d)), damage_max=float(jnp.max(d)),
                fixed_H_image_damage_min=inner['damage_min'], fixed_H_image_damage_max=inner['damage_max'],
                fixed_H_image_AT2_relative=inner['AT2_relative'],
                coupled_AT2_relative=at2_metrics(d, H, matvec, gc0, gc1),
                raw_current_H_AT2_relative=float(measure(net, H, load)['AT2_relative']),
                damage_change=float(jnp.max(abs(d-previous_d))),
                q_change_over_load=float(jnp.max(abs(q-previous_q)))/load_value,
                H_change_relative=float(jnp.max(abs(H-previous_H)))/scale,
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
            count, last_outer = outer['consecutive'], outer
            row['outer_convergence'] = outer
            atomic_json(folder/f'sweep_{sweep:04d}.json', row)
            print(json.dumps(dict(event='outer', index=index, sweep=sweep, reaction=reaction,
                current_AT2=row['coupled_AT2_relative'], raw_current_AT2=row['raw_current_H_AT2_relative'],
                damage_max=row['damage_max'], damage_change=row['damage_change'], ready=outer['converged'])), flush=True)
            save_pending(folder, net, q, d, H, index=index, load=load_value, completed_sweep=sweep,
                consecutive=count, reaction=reaction, mechanical=mechanical, parent_sha=parent_sha,
                last_image=dict(H_used=H_used, damage_image=damage_image))
            if outer['converged']:
                status = 'ready'
                break
        atomic_fields(folder/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
            H_used=np.asarray(H_used), damage_image=np.asarray(damage_image), d_parent=np.asarray(d_parent),
            H_parent=np.asarray(H_parent), raw_current_image=np.asarray(predict(net, H, load)))
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        force = mechanics.matrix(np.asarray(d))@np.asarray(q)
        host_force = float(np.linalg.norm(force[mechanics.free]))
        if host_force > mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('continuation independent host FEM force replay failed')
        report = dict(index=index, load=load_value, status=status, parent_sha256=parent_sha,
            state_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
            inner=inner, outer_convergence=last_outer, inner_calls=sweep,
            accepted_inner_sweeps=sweep if status=='ready' else sweep-1 if status=='inner_rejected' else sweep,
            reaction=reaction, seconds=time.perf_counter()-begin, host_force_norm=host_force,
            host_reaction=float(force@mechanics.load_direction), physical_state_promoted=status=='ready')
        atomic_json(folder/'result.json', report)
        curve.append(report)
        complete = status != 'ready' or index == c.continuation.last_index
        atomic_json(root/'result.json', dict(complete=complete, status=status if complete else 'running', curve=curve,
            inherited_ready_loads=parent_report['index'], new_ready_loads=sum(a['status']=='ready' for a in curve),
            total_ready_loads=parent_report['index']+sum(a['status']=='ready' for a in curve),
            requested_last_index=c.continuation.last_index, seconds=time.perf_counter()-started))
        print(json.dumps(dict(event='load_complete', index=index, status=status, reaction=reaction)), flush=True)
        if status != 'ready':
            break
        q_parent, d_parent, H_parent, parent_sha = q, d, H, report['state_sha256']
    if protected != {path: sha(path) for path in protected}:
        raise RuntimeError('bounded continuation protected sources/parents changed')
    return curve


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    c = runpy.run_path(args.config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c, resume=args.resume, parent_loader=verify_parent, setup_fn=setup,
        extra_code_names=[])
