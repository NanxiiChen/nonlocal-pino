"""Same-load recovery with additional inner budget and unchanged final gates."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.damage_range_tolerance import TolerantLocalMechanics
from examples.fracture2d._legacy.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d._legacy.compact_damage_loading import history_feature_input
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import load_geometry, at2_metrics, sha
from examples.fracture2d._legacy.outer_stopping import evaluate, validate_controls
from examples.fracture2d._legacy.output_damage_continue import atomic_fields, save_pending
from examples.fracture2d._legacy.plain_damage_head import make_problem, fresh_network
from examples.fracture2d._legacy.range_damage_optimization import (
    damage_diagnostics, make_augmented_objective, update_duals,
)
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def recovery_fields(source, mechanics, *, parent_index, completed_outer, load):
    parent_folder = source
    folder = source/f'state_{parent_index+1:04d}'
    parent = json.loads((parent_folder/'accepted_parent_result.json').read_text())
    failed = json.loads((folder/'result.json').read_text())
    meta = json.loads((folder/'pending.json').read_text())
    if (parent['status'] != 'ready' or not parent['physical_state_promoted']
            or not parent['outer_convergence']['converged']
            or failed['status'] != 'inner_rejected' or failed['physical_state_promoted']
            or meta['completed_sweep'] != completed_outer or meta['index'] != parent_index+1
            or meta['load'] != load or not meta['has_image']
            or failed['inner_calls'] != completed_outer+1
            or failed['accepted_inner_sweeps'] != completed_outer
            or meta['parent_sha256'] != parent['state_sha256']
            or failed['parent_sha256'] != parent['state_sha256']):
        raise ValueError('same-load recovery ancestry/counters/acceptance differ')
    for name in ('fields_file','network_file'):
        if Path(meta[name]).name != meta[name]:
            raise ValueError('same-load pending path must be a filename')
    for name, path in (('fields',folder/meta['fields_file']),('network',folder/meta['network_file'])):
        if sha(path) != meta[name+'_sha256']:
            raise ValueError('same-load pending checksum differs')
    if (parent['state_sha256'] != sha(source/'accepted_parent_fields.npz')
            or failed['state_sha256'] != sha(folder/'fields.npz')
            or failed['network_sha256'] != sha(folder/'network.eqx')):
        raise ValueError('same-load final state/network checksum differs')
    with np.load(source/'accepted_parent_fields.npz',allow_pickle=False) as f:
        parents = {k:f[k].copy() for k in ('q','d','H')}
    with np.load(folder/meta['fields_file'],allow_pickle=False) as f:
        pending = {k:f[k].copy() for k in ('q','d','H')}
    with np.load(folder/'fields.npz',allow_pickle=False) as f:
        for name in ('q','d','H'):
            np.testing.assert_array_equal(pending[name],f[name])
        np.testing.assert_array_equal(f['H_used'],pending['H'])
        np.testing.assert_array_equal(f['d_parent'],parents['d'])
        np.testing.assert_array_equal(f['H_parent'],parents['H'])
    for state in (parents,pending):
        for name in ('q','d','H'):
            shape = mechanics.load_direction.shape if name=='q' else (mechanics.n_material,)
            if state[name].shape != shape or state[name].dtype != np.float64 or not np.isfinite(state[name]).all():
                raise ValueError('same-load physical shape/dtype/finite values differ')
        if np.min(state['H'])<0:
            raise ValueError('same-load history must be nonnegative')
    np.testing.assert_array_equal(pending['H'],np.maximum(parents['H'],mechanics.driver(pending['q'])))
    np.testing.assert_array_equal(pending['q'][mechanics.constrained],load*mechanics.load_direction[mechanics.constrained])
    force = mechanics.matrix(pending['d'])@pending['q']
    if np.linalg.norm(force[mechanics.free])>meta['mechanical']['convergence_limit']*(1+1e-3):
        raise ValueError('same-load recovered FEM force is unready')
    np.testing.assert_allclose(float(force@mechanics.load_direction),meta['reaction'],rtol=1e-9,atol=1e-6)
    return parents,pending,meta


def validate_recovery_configuration(c, original):
    """Allow only declared budget, output and recovery metadata changes."""
    expected = c.to_dict()
    expected['continuation'] = original['config']['continuation']
    expected['range']['lbfgs']['maxiter'] = 2048
    expected['loading']['output_dir'] = str(Path(c.continuation.source_dir))
    if expected != original['config']:
        raise ValueError('budget recovery differs beyond declared budget/output/recovery metadata')


def run(c):
    validate_controls(c.outer_stop)
    indices = list(c.loading.indices)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or indices != list(range(53,76)) or c.data.load_step != .0001
            or c.cold.history_feature_scaling != 'physical' or c.stagger.omega != .5
            or not c.outer_stop.enabled or c.range.method != 'residual_l16'
            or list(c.loading.penalties) != [1e3,1e4,1e5,1e6]
            or c.range.lbfgs.maxiter != 8192 or c.range.lbfgs.residual_limit != .001
            or c.range.lbfgs.range_tolerance != 2e-6 or c.reporting.range_atol != 2e-6
            or c.continuation.range_atol != 2e-6
            or c.continuation.accepted_parent_index != 52 or c.continuation.completed_outer != 123):
        raise ValueError('budget recovery requires8192 iterations/stage, unchanged gates and explicit2e-6 range')
    source = Path(c.continuation.source_dir)
    required = {'identity.json','audit.json','result.json','parameterization.npz',
        'accepted_parent_result.json','accepted_parent_fields.npz',
        'state_0053/result.json','state_0053/fields.npz','state_0053/network.eqx',
        'state_0053/pending.json','state_0053/pending_0123_fields.npz','state_0053/pending_0123_network.eqx',
        'state_0053/inner_0124.json'}
    if set(c.continuation.source) != required:
        raise ValueError('linear recovery requires all certified accepted/pending/failed source pins')
    source_pins = {}
    for name,digest in c.continuation.source.items():
        path = source/name
        if sha(path) != digest:
            raise ValueError(f'linear recovery source differs: {path}')
        source_pins[str(path)] = digest
    original = json.loads((source/'identity.json').read_text())
    audit = json.loads((source/'audit.json').read_text())
    previous = json.loads((source/'result.json').read_text())
    if (not audit['pass_replay'] or not audit['complete'] or audit['accepted_new_loads'] != 0
            or audit['accepted_parent_index'] != 52 or audit['last_completed_outer'] != 123
            or audit['failed_inner'] != 124 or audit['physical_state_promoted']
            or not previous['complete'] or previous['ready_loads'] != 0
            or previous['status'] != 'inner_rejected'):
        raise ValueError('budget recovery requires certified final tolerant accuracy stop')
    validate_recovery_configuration(c, original)
    for path,digest in original['source'].items():
        if sha(path) != digest:
            raise ValueError('linear recovery original protected source changed')
        source_pins[path] = digest
    for name,digest in original['code'].items():
        if (sha(Path(__file__).with_name(name)) != digest
                or sha(source/'code_snapshot'/name) != digest):
            raise ValueError('linear recovery original numerical source changed')
    basis_path = Path(c.cold.basis_path)
    if sha(basis_path) != c.cold.basis_sha256:
        raise ValueError('cold trial compact geometry differs')
    original_mechanics,operator,geometry = load_geometry(c)
    mechanics = TolerantLocalMechanics(original_mechanics.coordinates,original_mechanics.triangles,
        original_mechanics.n_material,original_mechanics.n_material_cells,original_mechanics.volumes,
        E=c.physical.E,nu=c.physical.nu,k_res=c.physical.k_res,
        constrained=original_mechanics.constrained,load_direction=original_mechanics.load_direction,
        damage_range_atol=c.continuation.range_atol)
    fem_path = Path(c.trial.geometry_dir)/'fenics_parity.json'
    fem = json.loads(fem_path.read_text())
    if not fem['pass'] or fem['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('cold trial independent local-FEM validation missing')
    matrix = load_npz(basis_path)
    gc0,gc1 = c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    features,info = make_features(operator,mechanics.coordinates[:mechanics.n_material],
        gc0=gc0,length=c.physical.l0,tip=list(c.conditioned.tip),powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    def known_features(H,load):
        return features(history_feature_input(H,load,c.cold.reference_load,c.cold.history_feature_scaling))
    if matrix.shape[0] != mechanics.n_material:
        raise ValueError('cold compact geometry/material ordering differs')
    template = fresh_network(c,info['input_dimension'],matrix,head='identity')
    net = eqx.tree_deserialise_leaves(source/'state_0053/network.eqx',template)
    with np.load(source/'parameterization.npz',allow_pickle=False) as f:
        columns,gain = f['column_scale'].copy(),float(f['trunk_gain'])
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,
                                   maxiter=c.validation.cg_maxiter)
    parents,pending,recovery = recovery_fields(source,mechanics,
        parent_index=c.continuation.accepted_parent_index,
        completed_outer=c.continuation.completed_outer,load=indices[0]*c.data.load_step)
    q_parent,d_parent,H_parent = [jnp.asarray(parents[k]) for k in ('q','d','H')]
    index = indices[0]
    load_value,load = index*c.data.load_step,jnp.asarray(index*c.data.load_step)
    q,d,H = [jnp.asarray(pending[k]) for k in ('q','d','H')]
    reaction,mechanical = recovery['reaction'],recovery['mechanical']
    s = dict(matrix=matrix,operator=operator,reference_load=c.cold.reference_load,features=known_features)
    image,base_metrics,objectives,_ = make_problem(c,s,columns,gain,head='identity')
    matvec,_ = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    with np.load(Path(c.trial.geometry_dir)/'mesh.npz',allow_pickle=False) as f:
        volumes = jnp.asarray(f['volumes'])
    if volumes.shape != H.shape or float(volumes.min()) <= 0:
        raise ValueError('linear loading requires matching positive material volumes')
    thresholds = tuple(c.range.activity_thresholds)
    def field_metrics(damage,history):
        return damage_diagnostics(damage,history,matvec,gc0=gc0,gc1=gc1,
            volumes=volumes,thresholds=thresholds)
    def metrics(net,history,load):
        return field_metrics(image(net,history,load),history)
    predict,measure = eqx.filter_jit(image),eqx.filter_jit(metrics)
    protected = {**source_pins,str(basis_path):sha(basis_path)}
    for name in ('identity.json','mesh.npz','damage_operator.npz','fenics_parity.json'):
        path = Path(c.trial.geometry_dir)/name
        protected[str(path)] = sha(path)
    names = ['linear_damage_budget_continue.py','linear_damage_continue.py','damage_range_tolerance.py','linear_damage_loading.py','plain_damage_head.py','range_damage_optimization.py','plain_sigmoid_damage.py','cached_damage_optimization.py',
        'conditioned_damage.py','compact_damage_features.py','compact_damage_loading.py',
        'local_damage_trial.py','local_mechanics.py','outer_stopping.py','split_pinn_gradients.py',
        'output_damage_continue.py','train.py']
    code = {name:sha(Path(__file__).with_name(name)) for name in names}
    root = Path(c.loading.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    identity = dict(schema='linear-damage-additional-budget-continuation-v1',config=c.to_dict(),source=protected,
        code=code,geometry=geometry,features=info,own_zero_initial_state=False,inherited_zero_start_ancestry=True,
        range_atol=c.continuation.range_atol,additional_inner_budget=True,
        prior_maxiter=2048,maxiter_per_stage=c.range.lbfgs.maxiter,
        residual_accuracy_unchanged=True,optimizer_memory_restarted=True,AL_duals_restarted=True,
        recovered_outer=c.continuation.completed_outer,
        accepted_parent_index=c.continuation.accepted_parent_index,physical_parent_promoted=False,
        fresh_network=False,own_zero_fresh_network_ancestry=True,own_H_parameter_seed=True,
        old_head_weights=False,direct_targets=False,damage_correction=False,
        frozen_reference_in_output=False,reference_load=c.cold.reference_load,
        trainable_parameters=sum(a.size for a in jax.tree.leaves(eqx.filter(net,eqx.is_inexact_array))))
    atomic_json(root/'identity.json',identity)
    (root/'code_snapshot').mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    np.savez_compressed(root/'parameterization.npz',column_scale=columns,trunk_gain=gain)
    eqx.tree_serialise_leaves(root/'initial_network.eqx',net)
    shutil.copyfile(source/'accepted_parent_fields.npz',root/'accepted_parent_fields.npz')
    shutil.copyfile(source/'accepted_parent_result.json',root/'accepted_parent_result.json')
    atomic_fields(root/'initial_physical_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),
        d_parent=np.asarray(d_parent),H_parent=np.asarray(H_parent))
    parent_sha = sha(root/'accepted_parent_fields.npz')
    def augmented_metrics(net,context,load):
        return metrics(net,context[0],load)
    augmented = make_augmented_objective(image,objectives[c.range.method],gc0=gc0)
    fit_augmented = make_cached_lbfgs(net,augmented,augmented_metrics)[0]
    curve = []
    for index in indices:
        load_value = index*c.data.load_step
        load = jnp.asarray(load_value,jnp.float64)
        if index == indices[0]:
            start_sweep = c.continuation.completed_outer+1
        else:
            start_sweep = 1
            d = d_parent
            q,reaction,mechanical = solve_q(d,load,q_parent)
            H = jnp.maximum(H_parent,jnp.asarray(mechanics.driver(np.asarray(q))))
        folder = root/f'state_{index:04d}'
        folder.mkdir()
        started = time.perf_counter()
        if index == indices[0]:
            first = {k:float(v) for k,v in measure(net,H,load).items()}
            failed = json.loads((source/'state_0053/result.json').read_text())
            if not np.isfinite(list(first.values())).all():
                raise FloatingPointError('budget recovery initial NN diagnostics are nonfinite')
            for name in ('AT2_relative','damage_min','damage_max'):
                np.testing.assert_allclose(first[name],failed['inner'][name],rtol=1e-10,atol=1e-12)
            atomic_json(root/'initial_fixed_H_result.json',dict(metrics=first,
                unchanged_accuracy_limit=c.range.lbfgs.residual_limit,
                range_atol=c.range.lbfgs.range_tolerance))
        save_pending(folder,net,q,d,H,index=index,load=load_value,completed_sweep=start_sweep-1,
            consecutive=0,reaction=reaction,mechanical=mechanical,parent_sha=parent_sha,last_image=None)
        status,count = 'stagger_budget_exhausted',0
        last_outer = (json.loads((source/'state_0053/result.json').read_text())['outer_convergence']
                      if index == indices[0] else None)
        for sweep in range(start_sweep,c.outer_stop.max_sweeps+1):
            previous_q,previous_d,previous_H,previous_R = q,d,H,reaction
            H_used = jax.lax.stop_gradient(H)
            stages = []
            lower,upper = jnp.zeros_like(H_used),jnp.zeros_like(H_used)
            for inner_stage,penalty_value in enumerate(c.loading.penalties,start=1):
                context = (H_used,lower,upper,jnp.asarray(penalty_value,jnp.float64))
                def progress(row):
                    print(json.dumps(dict(event='inner',index=index,sweep=sweep,
                        stage=inner_stage,penalty=penalty_value,**row)),flush=True)
                net,optimization = fit_augmented(net,context,load,c.range.lbfgs.to_dict(),progress=progress)
                damage_candidate = predict(net,H_used,load)
                lower,upper = update_duals(damage_candidate,H_used,lower,upper,
                    context[3],gc0=gc0)
                stages.append(dict(stage=inner_stage,penalty=penalty_value,**optimization))
                if optimization['status'] == 'residual_reached':
                    break
            damage_image = predict(net,H_used,load)
            inner = {k:float(v) for k,v in measure(net,H_used,load).items()}
            atomic_json(folder/f'inner_{sweep:04d}.json',dict(stages=stages,metrics=inner))
            if index == indices[0] and sweep == start_sweep:
                atomic_fields(root/'fixed_H_fields.npz',damage=np.asarray(damage_image),H_used=np.asarray(H_used))
                eqx.tree_serialise_leaves(root/'fixed_H_network.eqx',net)
                atomic_json(root/'fixed_H_result.json',dict(ready=optimization['status']=='residual_reached',
                    metrics=inner,stages=stages,seconds=time.perf_counter()-started))
            if optimization['status'] != 'residual_reached':
                status = 'inner_rejected'
                break
            d = d+c.stagger.omega*(damage_image-d)
            q,reaction,mechanical = solve_q(d,load,q)
            H = jnp.maximum(H_parent,jnp.asarray(mechanics.driver(np.asarray(q))))
            scale = max(float(jnp.max(abs(H))),float(jnp.max(abs(previous_H))),c.outer_stop.scale_floor)
            ratio = np.linalg.norm(np.asarray(q).reshape(-1,2)[:mechanics.n_material],axis=1)/load_value
            row = dict(index=index,load=load_value,sweep=sweep,reaction=reaction,
                damage_min=float(d.min()),damage_max=float(d.max()),
                fixed_H_image_damage_min=inner['damage_min'],fixed_H_image_damage_max=inner['damage_max'],
                fixed_H_image_AT2_relative=inner['AT2_relative'],coupled_AT2_relative=at2_metrics(d,H,matvec,gc0,gc1),
                raw_current_H_AT2_relative=float(measure(net,H,load)['AT2_relative']),
                damage_change=float(jnp.max(abs(d-previous_d))),q_change_over_load=float(jnp.max(abs(q-previous_q)))/load_value,
                H_change_relative=float(jnp.max(abs(H-previous_H)))/scale,
                reaction_change_relative=abs(reaction-previous_R)/max(abs(reaction),abs(previous_R),c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(d_parent-d,0.))),history_decrease=float(jnp.max(jnp.maximum(H_parent-H,0.))),
                drift_count=int(np.count_nonzero(ratio>direct.DRIFT_ENVELOPE)),mechanics=mechanical,parent_sha256=parent_sha)
            physical_stats = {k:float(v) for k,v in field_metrics(d,H).items()}
            row['physical_diagnostics'] = physical_stats
            row['raw_neural_diagnostics'] = {k:float(v) for k,v in measure(net,H,load).items()}
            outer = evaluate(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count)
            raw_pass = row['raw_current_H_AT2_relative']<=c.reporting.coupled_AT2_relative_limit
            outer['checks']['raw_current_H_AT2'] = dict(value=row['raw_current_H_AT2_relative'],
                limit=c.reporting.coupled_AT2_relative_limit,**{'pass':raw_pass})
            raw_stats = row['raw_neural_diagnostics']
            lower_pass = raw_stats['damage_min'] >= -c.range.lbfgs.range_tolerance
            upper_pass = raw_stats['damage_max'] <= 1+c.range.lbfgs.range_tolerance
            outer['checks']['raw_current_H_damage_min'] = dict(value=raw_stats['damage_min'],
                limit=-c.range.lbfgs.range_tolerance,**{'pass':lower_pass})
            outer['checks']['raw_current_H_damage_max'] = dict(value=raw_stats['damage_max'],
                limit=1+c.range.lbfgs.range_tolerance,**{'pass':upper_pass})
            if not (raw_pass and lower_pass and upper_pass):
                outer.update(pass_this_sweep=False,consecutive=0,converged=False,
                    reason='raw_neural_current_H_rejected')
            count,last_outer = outer['consecutive'],outer
            row['outer_convergence'] = outer
            atomic_json(folder/f'sweep_{sweep:04d}.json',row)
            print(json.dumps(dict(event='outer',index=index,sweep=sweep,reaction=reaction,
                current_AT2=row['coupled_AT2_relative'],damage_change=row['damage_change'],ready=outer['converged'])),flush=True)
            save_pending(folder,net,q,d,H,index=index,load=load_value,completed_sweep=sweep,
                consecutive=count,reaction=reaction,mechanical=mechanical,parent_sha=parent_sha,
                last_image=dict(H_used=H_used,damage_image=damage_image))
            if index == indices[0] and sweep == start_sweep:
                snapshot = root/f'recovery_snapshot_{sweep:04d}'
                snapshot.mkdir()
                shutil.copyfile(folder/'pending.json',snapshot/'pending.json')
                manifest = json.loads((folder/'pending.json').read_text())
                for key in ('fields_file','network_file'):
                    shutil.copyfile(folder/manifest[key],snapshot/manifest[key])
                atomic_json(snapshot/'sweep.json',row)
            if outer['converged']:
                status = 'ready'
                break
        atomic_fields(folder/'fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),H_used=np.asarray(H_used),
            damage_image=np.asarray(damage_image),raw_current_image=np.asarray(predict(net,H,load)),
            d_parent=np.asarray(d_parent),H_parent=np.asarray(H_parent))
        eqx.tree_serialise_leaves(folder/'network.eqx',net)
        force = mechanics.matrix(np.asarray(d))@np.asarray(q)
        host_force = float(np.linalg.norm(force[mechanics.free]))
        if host_force>mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('cold-trial independent host FEM force failed')
        report = dict(index=index,load=load_value,status=status,inner=inner,outer_convergence=last_outer,
            inner_calls=sweep,accepted_inner_sweeps=sweep-1 if status=='inner_rejected' else sweep,
            resumed_from_outer=start_sweep-1,completed_new_outers=sweep-start_sweep if status=='inner_rejected' else sweep-start_sweep+1,
            physical_state_promoted=status=='ready',parent_sha256=parent_sha,
            state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'),
            reaction=reaction,host_reaction=float(force@mechanics.load_direction),host_force_norm=host_force,
            seconds=time.perf_counter()-started)
        atomic_json(folder/'result.json',report)
        curve.append(report)
        atomic_json(root/'result.json',dict(complete=False,status=status,curve=curve,
            ready_loads=sum(row['status']=='ready' for row in curve),
            requested_indices=indices,own_zero_initial_state=False,inherited_zero_start_ancestry=True,range_atol=c.continuation.range_atol))
        if status != 'ready':
            break
        q_parent,d_parent,H_parent = q,d,H
        parent_sha = report['state_sha256']
        if index == 53:
            print(json.dumps(dict(event='critical_load_passed',ready_loads=1)),flush=True)
    if protected!={path:sha(path) for path in protected} or code!={name:sha(Path(__file__).with_name(name)) for name in names}:
        raise RuntimeError('cold-trial protected geometry/source/code changed')
    print(json.dumps(dict(event='complete',status=status,index=index,reaction=reaction,seconds=report['seconds'])),flush=True)
    atomic_json(root/'result.json',dict(complete=True,status=status,curve=curve,
        ready_loads=sum(row['status']=='ready' for row in curve),
        requested_indices=indices,own_zero_initial_state=False,inherited_zero_start_ancestry=True,range_atol=c.continuation.range_atol))
    return curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    c = runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    run(c)
