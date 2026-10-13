"""Accepted-only own-zero loading with a certified linear damage network."""
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

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d.compact_damage_loading import history_feature_input
from examples.fracture2d.conditioned_damage import make_features
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import load_geometry, at2_metrics, sha
from examples.fracture2d.outer_stopping import evaluate, validate_controls
from examples.fracture2d.output_damage_continue import atomic_fields, save_pending
from examples.fracture2d.plain_damage_head import make_problem, fresh_network
from examples.fracture2d.range_damage_optimization import (
    damage_diagnostics, make_augmented_objective, update_duals,
)
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def run(c):
    validate_controls(c.outer_stop)
    indices = list(c.loading.indices)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or indices != list(range(1,76)) or c.data.load_step != .0001
            or c.cold.history_feature_scaling != 'physical' or c.stagger.omega != .5
            or not c.outer_stop.enabled or c.range.method != 'residual_l16'
            or list(c.loading.penalties) != [1e3,1e4,1e5,1e6]
            or c.range.lbfgs.maxiter != 2048 or c.range.lbfgs.residual_limit != .001
            or c.range.lbfgs.range_tolerance != 1e-10):
        raise ValueError('linear loading requires own-zero original schedule/gates and declared AL controls')
    source = Path(c.loading.source_dir)
    required = {'identity.json','audit.json','own_fields.npz','parameterization.npz',
        'AL_0002/network.eqx','AL_0002/fields.npz','result.json'}
    if set(c.loading.source) != required:
        raise ValueError('linear loading requires all certified fixed-H source pins')
    source_pins = {}
    for name,digest in c.loading.source.items():
        path = source/name
        if sha(path) != digest:
            raise ValueError(f'linear loading source differs: {path}')
        source_pins[str(path)] = digest
    original = json.loads((source/'identity.json').read_text())
    audit = json.loads((source/'audit.json').read_text())
    previous = json.loads((source/'result.json').read_text())
    if (not audit['pass_replay'] or not audit['records'][-1]['ready']
            or not previous['complete'] or not previous['fixed_H_ready']
            or previous['physical_state_promoted'] or not original['no_direct_targets']
            or not original['no_output_projection']):
        raise ValueError('linear loading requires certified ready own-H parameters only')
    for key in ('physical','geometry','nl','trial','conditioned','validation','arch','train',
                'head','cold','range','extension'):
        if original['config'][key] != c[key].to_dict():
            raise ValueError(f'linear loading model/representation differs: {key}')
    basis_path = Path(c.cold.basis_path)
    if sha(basis_path) != c.cold.basis_sha256:
        raise ValueError('cold trial compact geometry differs')
    mechanics,operator,geometry = load_geometry(c)
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
    net = eqx.tree_deserialise_leaves(source/'AL_0002/network.eqx',template)
    with np.load(source/'parameterization.npz',allow_pickle=False) as f:
        columns,gain = f['column_scale'].copy(),float(f['trunk_gain'])
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,
                                   maxiter=c.validation.cg_maxiter)
    q_parent = jnp.zeros(len(mechanics.load_direction),jnp.float64)
    d_parent,H_parent = [jnp.zeros(mechanics.n_material,jnp.float64) for _ in range(2)]
    index = indices[0]
    load_value,load = index*c.data.load_step,jnp.asarray(index*c.data.load_step)
    d = d_parent
    q,reaction,mechanical = solve_q(d,load,q_parent)
    H = jnp.maximum(H_parent,jnp.asarray(mechanics.driver(np.asarray(q))))
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
    names = ['linear_damage_loading.py','plain_damage_head.py','range_damage_optimization.py','plain_sigmoid_damage.py','cached_damage_optimization.py',
        'conditioned_damage.py','compact_damage_features.py','compact_damage_loading.py',
        'local_damage_trial.py','local_mechanics.py','outer_stopping.py','split_pinn_gradients.py',
        'output_damage_continue.py','train.py']
    code = {name:sha(Path(__file__).with_name(name)) for name in names}
    root = Path(c.loading.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    identity = dict(schema='linear-damage-own-zero-loading-v1',config=c.to_dict(),source=protected,
        code=code,geometry=geometry,features=info,own_zero_initial_state=True,
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
    atomic_fields(root/'zero_fields.npz',q=np.asarray(q_parent),d=np.asarray(d_parent),H=np.asarray(H_parent))
    atomic_fields(root/'initial_physical_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H))
    parent_sha = sha(root/'zero_fields.npz')
    def augmented_metrics(net,context,load):
        return metrics(net,context[0],load)
    augmented = make_augmented_objective(image,objectives[c.range.method],gc0=gc0)
    fit_augmented = make_cached_lbfgs(net,augmented,augmented_metrics)[0]
    curve = []
    for index in indices:
        load_value = index*c.data.load_step
        load = jnp.asarray(load_value,jnp.float64)
        d = d_parent
        q,reaction,mechanical = solve_q(d,load,q_parent)
        H = jnp.maximum(H_parent,jnp.asarray(mechanics.driver(np.asarray(q))))
        folder = root/f'state_{index:04d}'
        folder.mkdir()
        started = time.perf_counter()
        if index == indices[0]:
            first = {k:float(v) for k,v in measure(net,H,load).items()}
            if (first['AT2_relative'] > .001 or first['damage_min'] < -1e-10
                    or first['damage_max'] > 1+1e-10):
                raise ValueError('own-zero first load did not reproduce certified fixed-H readiness')
        save_pending(folder,net,q,d,H,index=index,load=load_value,completed_sweep=0,
            consecutive=0,reaction=reaction,mechanical=mechanical,parent_sha=parent_sha,last_image=None)
        status,count,last_outer = 'stagger_budget_exhausted',0,None
        for sweep in range(1,c.outer_stop.max_sweeps+1):
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
            if index == indices[0] and sweep == 1:
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
            physical_state_promoted=status=='ready',parent_sha256=parent_sha,
            state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'),
            reaction=reaction,host_reaction=float(force@mechanics.load_direction),host_force_norm=host_force,
            seconds=time.perf_counter()-started)
        atomic_json(folder/'result.json',report)
        curve.append(report)
        atomic_json(root/'result.json',dict(complete=False,status=status,curve=curve,
            ready_loads=sum(row['status']=='ready' for row in curve),
            requested_indices=indices,own_zero_initial_state=True))
        if status != 'ready':
            break
        q_parent,d_parent,H_parent = q,d,H
        parent_sha = report['state_sha256']
        if index == 10:
            print(json.dumps(dict(event='short_prefix_passed',ready_loads=10)),flush=True)
    if protected!={path:sha(path) for path in protected} or code!={name:sha(Path(__file__).with_name(name)) for name in names}:
        raise RuntimeError('cold-trial protected geometry/source/code changed')
    print(json.dumps(dict(event='complete',status=status,index=index,reaction=reaction,seconds=report['seconds'])),flush=True)
    atomic_json(root/'result.json',dict(complete=True,status=status,curve=curve,
        ready_loads=sum(row['status']=='ready' for row in curve),
        requested_indices=indices,own_zero_initial_state=True))
    return curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    c = runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    run(c)
