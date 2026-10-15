"""Accepted-only gradual loading of a fresh plain-sigmoid damage network."""
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
from examples.fracture2d._legacy.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d._legacy.compact_damage_loading import history_feature_input
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import load_geometry, at2_metrics, sha
from examples.fracture2d._legacy.outer_stopping import evaluate, validate_controls
from examples.fracture2d._legacy.output_damage_continue import atomic_fields, save_pending
from examples.fracture2d._legacy.plain_sigmoid_damage import make_gradual_problem
from examples.fracture2d._legacy.plain_sigmoid_cold import fresh_network, initial_residual_scales
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def run(c):
    validate_controls(c.outer_stop)
    indices = list(c.cold.indices)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or indices != list(range(1,76)) or c.data.load_step != .0001
            or c.cold.reference_load != indices[0]*c.data.load_step
            or not 0 < c.cold.initial_damage < 1 or c.stagger.omega != .5
            or not c.outer_stop.enabled or list(c.cold.methods) != ['residual_l32','residual_l16']
            or c.cold.parameterization != 'unscaled_logits_initial_AT2_jacobian'
            or c.cold.history_feature_scaling != 'physical'
            or c.cold.initial_damage != 1e-5
            or c.cold.trunk_gain != 1.
            or c.cold.lbfgs.residual_limit != c.outer_stop.inner_AT2_relative_limit):
        raise ValueError('gradual trial requires explicit own-zero1..75 loads and original gates')
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
    net = fresh_network(c,info['input_dimension'],matrix)
    gain = c.cold.trunk_gain
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,
                                   maxiter=c.validation.cg_maxiter)
    q_parent = jnp.zeros(len(mechanics.load_direction),jnp.float64)
    d_parent,H_parent = [jnp.zeros(mechanics.n_material,jnp.float64) for _ in range(2)]
    index = indices[0]
    load_value,load = index*c.data.load_step,jnp.asarray(index*c.data.load_step)
    d = d_parent
    q,reaction,mechanical = solve_q(d,load,q_parent)
    H = jnp.maximum(H_parent,jnp.asarray(mechanics.driver(np.asarray(q))))
    columns = initial_residual_scales(matrix,operator.to_scipy(),np.asarray(H),
        initial_damage=c.cold.initial_damage,gc0=gc0,gc1=gc1)
    s = dict(matrix=matrix,operator=operator,reference_load=c.cold.reference_load,features=known_features)
    image,metrics,objectives,_ = make_gradual_problem(c,s,columns,gain)
    predict,measure = eqx.filter_jit(image),eqx.filter_jit(metrics)
    protected = {str(basis_path):sha(basis_path)}
    for name in ('identity.json','mesh.npz','damage_operator.npz','fenics_parity.json'):
        path = Path(c.trial.geometry_dir)/name
        protected[str(path)] = sha(path)
    names = ['plain_sigmoid_gradual.py','plain_sigmoid_cold.py','plain_sigmoid_damage.py','cached_damage_optimization.py',
        'conditioned_damage.py','compact_damage_features.py','compact_damage_loading.py',
        'local_damage_trial.py','local_mechanics.py','outer_stopping.py','split_pinn_gradients.py',
        'output_damage_continue.py','train.py']
    code = {name:sha(Path(__file__).with_name(name)) for name in names}
    root = Path(c.cold.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    identity = dict(schema='plain-sigmoid-own-zero-gradual-v1',config=c.to_dict(),source=protected,
        code=code,geometry=geometry,features=info,own_zero_initial_state=True,
        fresh_network=True,old_weights=False,direct_targets=False,damage_correction=False,
        frozen_reference_in_output=False,reference_load=c.cold.reference_load,
        trainable_parameters=sum(a.size for a in jax.tree.leaves(eqx.filter(net,eqx.is_inexact_array))))
    atomic_json(root/'identity.json',identity)
    (root/'code_snapshot').mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    np.savez_compressed(root/'parameterization.npz',column_scale=columns,trunk_gain=gain)
    eqx.tree_serialise_leaves(root/'initial_network.eqx',net)
    matvec,_ = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    atomic_fields(root/'zero_fields.npz',q=np.asarray(q_parent),d=np.asarray(d_parent),H=np.asarray(H_parent))
    atomic_fields(root/'initial_physical_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H))
    parent_sha = sha(root/'zero_fields.npz')
    fitters = {method:make_cached_lbfgs(net,objectives[method],metrics)[0] for method in c.cold.methods}
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
            np.testing.assert_allclose(predict(net,H,load),c.cold.initial_damage,rtol=1e-12,atol=1e-14)
        save_pending(folder,net,q,d,H,index=index,load=load_value,completed_sweep=0,
            consecutive=0,reaction=reaction,mechanical=mechanical,parent_sha=parent_sha,last_image=None)
        status,count,last_outer = 'stagger_budget_exhausted',0,None
        for sweep in range(1,c.outer_stop.max_sweeps+1):
            previous_q,previous_d,previous_H,previous_R = q,d,H,reaction
            H_used = jax.lax.stop_gradient(H)
            stages = []
            for method in c.cold.methods:
                def progress(row):
                    print(json.dumps(dict(event='inner',index=index,sweep=sweep,method=method,**row)),flush=True)
                net,optimization = fitters[method](net,H_used,load,c.cold.lbfgs.to_dict(),progress=progress)
                stages.append(dict(method=method,**optimization))
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
            outer = evaluate(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count)
            raw_pass = row['raw_current_H_AT2_relative']<=c.reporting.coupled_AT2_relative_limit
            outer['checks']['raw_current_H_AT2'] = dict(value=row['raw_current_H_AT2_relative'],
                limit=c.reporting.coupled_AT2_relative_limit,**{'pass':raw_pass})
            if not raw_pass:
                outer.update(pass_this_sweep=False,consecutive=0,converged=False,reason='raw_neural_current_H_rejected')
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
