"""Explicit user-provisional loading with original signed FEM/nonlocal AT2 NN."""
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

from examples.fracture2d.asymmetric_damage_range import AsymmetricLocalMechanics,make_asymmetric_lbfgs,evaluate_asymmetric
from examples.fracture2d.core_halo_damage_trial import prepare_trial
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.provisional_damage import make_dynamic_problem,advancement,mesh_adjacency,crack_extent
from examples.fracture2d.range_damage_optimization import damage_diagnostics,update_duals


def run(c):
    a=c.core.provisional
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform!='gpu' or a.user_authorized is not True
            or a.initial_index!=53 or list(a.indices)!=list(range(54,76)) or c.stagger.omega!=.5
            or any(type(x) is not int or x<1 for x in [a.outer_updates,a.inner_updates])
            or a.outer_updates>c.outer_stop.max_sweeps or list(a.range_penalties)!=list(c.loading.penalties)
            or a.drift_envelope!=1.08):
        raise ValueError('provisional branch requires explicit user authorization and original fixed-shape CUDA model')
    p=prepare_trial(c);original=p.mechanics
    m=AsymmetricLocalMechanics(original.coordinates,original.triangles,original.n_material,original.n_material_cells,
        original.volumes,E=c.physical.E,nu=c.physical.nu,k_res=c.physical.k_res,
        constrained=original.constrained,load_direction=original.load_direction,
        negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
    pins={a.source_fields:a.source_fields_sha256,a.source_network:a.source_network_sha256,
        a.source_network_fields:a.source_network_fields_sha256}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError(f'provisional source hash differs:{path}')
    audit=json.loads(Path(a.source_network_audit).read_text())
    record=next(row for row in audit['polishes'] if row['mode']=='sampled_polish')
    if (not audit['pass_replay'] or record['network_sha256']!=a.source_network_sha256
            or record['fields_sha256']!=a.source_network_fields_sha256):raise ValueError('polished NN source audit differs')
    source_report=json.loads(Path(a.source_report).read_text())
    source_case=next(row for row in source_report['cases'] if row['mode']=='relaxed_candidate')
    if (not source_report['pass_own_FEM_audit'] or source_report['load']!=a.initial_index*c.data.load_step
            or source_case['fields_sha256']!=a.source_fields_sha256 or source_case['omega']!=c.stagger.omega):
        raise ValueError('provisional source must be audited original-relaxation candidate')
    with np.load(a.source_fields) as f:q,d,H=[f[key].copy() for key in ['q','d','H']]
    with np.load(a.source_network_fields) as f:saved_image,saved_H=f['d'].copy(),f['H'].copy()
    net=eqx.tree_deserialise_leaves(a.source_network,p.initial)
    with np.load(Path(c.loading.output_dir)/'accepted_parent_fields.npz') as f:H52=f['H'].copy()
    m.factors(d)
    np.testing.assert_array_equal(H,np.maximum(H52,m.driver(q)))
    np.testing.assert_array_equal(q[m.constrained],a.initial_index*c.data.load_step*m.load_direction[m.constrained])
    force=m.matrix(d)@q;reaction=float(force@m.load_direction)
    np.testing.assert_allclose(reaction,source_case['reaction'],rtol=1e-9,atol=1e-6)
    if np.linalg.norm(force[m.free])>source_case['host_free_force_limit']*(1+1e-3):raise ValueError('source force equilibrium fails')
    gc0,gc1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    predict,objective=make_dynamic_problem(p.bc,p.bv,p.matvec,gc0=gc0,gc1=gc1,power=c.core.power,
        floor=c.conditioned.loss_floor,trunk_gain=p.gain)
    predict=eqx.filter_jit(predict);feature=eqx.filter_jit(p.known_features)
    volumes=jnp.asarray(m.volumes);thresholds=tuple(c.range.activity_thresholds)
    def field_metrics(d,H):return damage_diagnostics(d,H,p.matvec,gc0=gc0,gc1=gc1,volumes=volumes,thresholds=thresholds)
    field_metrics=eqx.filter_jit(field_metrics)
    def metrics(net,context,load):return field_metrics(predict(net,context,load),context['H'])
    measure=eqx.filter_jit(metrics)
    zeros=jnp.zeros(p.n,jnp.float64)
    def context(history,load,lower,upper,penalty):
        h=jax.lax.stop_gradient(jnp.asarray(history))
        return dict(H=h,features=jax.lax.stop_gradient(feature(h,load)),lower=lower,upper=upper,penalty=jnp.asarray(penalty,jnp.float64))
    old_context=context(saved_H,jnp.asarray(a.initial_index*c.data.load_step),zeros,zeros,a.range_penalties[0])
    np.testing.assert_allclose(predict(net,old_context,jnp.asarray(.0053)),saved_image,rtol=1e-10,atol=1e-12)
    v0,g0=eqx.filter_value_and_grad(objective)(net,old_context,jnp.asarray(.0053))
    v1,g1=eqx.filter_value_and_grad(p.cached_objective)(net,p.full_context,jnp.asarray(.0053))
    np.testing.assert_allclose(v0,v1,rtol=1e-10,atol=1e-12)
    for x,y in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)):np.testing.assert_allclose(x,y,rtol=1e-8,atol=1e-9)
    fit=make_asymmetric_lbfgs(net,objective,metrics)[0]
    solve=m.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,maxiter=c.validation.cg_maxiter)
    graph=mesh_adjacency(m.triangles[:m.n_material_cells],population_size=m.n_material)
    def extent(damage):return crack_extent(damage,p.points,graph,threshold=a.crack_threshold,tip=a.tip,
        tip_radius=a.tip_radius,right_edge=a.right_edge,tolerance=a.coordinate_tolerance)
    def range_pass(values):return values['damage_min']>=-c.reporting.negative_range_atol and values['damage_max']<=1+c.reporting.positive_range_atol
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=False)
    names=['provisional_damage_loading.py','provisional_damage.py','core_halo_damage_trial.py','core_halo_damage.py',
        'asymmetric_damage_range.py','local_mechanics.py','outer_stopping.py','range_damage_optimization.py']
    code={name:sha(Path(__file__).with_name(name)) for name in names}
    atomic_json(root/'identity.json',dict(config=c.to_dict(),geometry=p.geometry,code=code,source_pins=pins,
        user_authorized_provisional_advancement=True,original_strict_branch_modified=False,
        accepted_strict_prefix=52,initial_user_provisional_index=53,source_outer=192,
        source_has_provisional_ancestry=True,certified_path=False,direct_training_targets=False,
        damage_output_clipped=False,physics_unchanged=True,dynamic_full_H_feature_cache=True))
    (root/'code_snapshot').mkdir()
    for name in names:shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    shutil.copyfile(a.source_report,root/'source_physical_report.json')
    np.savez_compressed(root/'initial_fields.npz',q=q,d=d,H=H)
    eqx.tree_serialise_leaves(root/'initial_network.eqx',net)
    folder=root/f'state_{a.initial_index:04d}';folder.mkdir()
    np.savez_compressed(folder/'fields.npz',q=q,d=d,H=H,H_training=saved_H,damage_image=saved_image,H_parent=H52)
    eqx.tree_serialise_leaves(folder/'network.eqx',net)
    initial=dict(index=53,load=.0053,reaction=reaction,source_case=source_case,
        **advancement(user_authorized=True,strict_ready=False,numerically_valid=True,provisional_ancestry=True),
        crack=extent(d),state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'))
    atomic_json(folder/'result.json',initial);curve=[initial];parent_sha=initial['state_sha256']
    q,d,H=jnp.asarray(q),jnp.asarray(d),jnp.asarray(H)
    def publish(complete,status):atomic_json(root/'result.json',dict(complete=complete,status=status,curve=curve,
        requested_indices=list(a.indices),all_states_locally_strict=all(row['local_strict_ready'] for row in curve),
        certified_path=False,user_provisional_ancestry=True))
    publish(False,'running');print(json.dumps(dict(event='provisional_source_committed',index=53,reaction=reaction)),flush=True)
    for index in a.indices:
        load=jnp.asarray(index*c.data.load_step,jnp.float64);folder=root/f'state_{index:04d}';folder.mkdir()
        d_parent,H_parent,q_parent=d,H,q;parent_H=H;started=time.perf_counter();count=0
        q,reaction,mechanical=solve(d,load,q)
        H=jnp.maximum(H_parent,jnp.asarray(m.driver(np.asarray(q))))
        for sweep in range(1,a.outer_updates+1):
            previous_q,previous_d,previous_H,previous_R=q,d,H,reaction
            H_used=jax.lax.stop_gradient(H);lower,upper=zeros,zeros;stages=[]
            for stage,penalty in enumerate(a.range_penalties,1):
                ctx=context(H_used,load,lower,upper,penalty)
                controls=c.range.lbfgs.to_dict();controls['maxiter']=a.inner_updates
                net,optimization=fit(net,ctx,load,controls,
                    progress=lambda row:print(json.dumps(dict(event='inner',index=index,sweep=sweep,stage=stage,**row)),flush=True))
                image=predict(net,ctx,load);inner={key:float(value) for key,value in measure(net,ctx,load).items()}
                stages.append(dict(stage=stage,penalty=penalty,**optimization))
                if range_pass(inner):break
                lower,upper=update_duals(image,H_used,lower,upper,ctx['penalty'],gc0=gc0)
            if not range_pass(inner):
                atomic_json(folder/'numerical_rejection.json',dict(index=index,sweep=sweep,reason='inner_range_rejected',inner=inner,stages=stages))
                publish(True,'numerical_rejected');return
            d=d+c.stagger.omega*(image-d)
            q,reaction,mechanical=solve(d,load,q)
            H=jnp.maximum(H_parent,jnp.asarray(m.driver(np.asarray(q))))
            physical={key:float(value) for key,value in field_metrics(d,H).items()}
            current_context=context(H,load,zeros,zeros,a.range_penalties[0])
            raw={key:float(value) for key,value in measure(net,current_context,load).items()}
            if not np.isfinite(list(physical.values())+list(raw.values())).all():raise FloatingPointError('nonfinite provisional fields')
            row=dict(index=index,load=float(load),sweep=sweep,reaction=reaction,
                damage_min=physical['damage_min'],damage_max=physical['damage_max'],
                fixed_H_image_damage_min=inner['damage_min'],fixed_H_image_damage_max=inner['damage_max'],
                fixed_H_image_AT2_relative=inner['AT2_relative'],coupled_AT2_relative=physical['AT2_relative'],
                raw_current_H_AT2_relative=raw['AT2_relative'],
                damage_change=float(jnp.max(abs(d-previous_d))),q_change_over_load=float(jnp.max(abs(q-previous_q)))/float(load),
                H_change_relative=float(jnp.max(abs(H-previous_H)))/max(float(H.max()),float(previous_H.max()),c.outer_stop.scale_floor),
                reaction_change_relative=abs(reaction-previous_R)/max(abs(reaction),abs(previous_R),c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(d_parent-d,0))),history_decrease=float(jnp.max(jnp.maximum(H_parent-H,0))),
                drift_count=int(np.count_nonzero(np.linalg.norm(np.asarray(q).reshape(-1,2)[:m.n_material],axis=1)/float(load)>a.drift_envelope)),
                mechanics=mechanical,parent_sha256=parent_sha,physical_diagnostics=physical,raw_neural_diagnostics=raw,
                inner_stages=stages,crack=extent(np.asarray(d)),user_provisional_ancestry=True)
            outer=evaluate_asymmetric(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count,
                negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
            outer['checks']['raw_current_H_AT2']=dict(value=raw['AT2_relative'],limit=c.reporting.coupled_AT2_relative_limit,
                **{'pass':raw['AT2_relative']<=c.reporting.coupled_AT2_relative_limit})
            outer['checks']['raw_current_H_range']=dict(value=max(0.,-raw['damage_min']),
                limit=c.reporting.negative_range_atol,**{'pass':range_pass(raw)})
            passed=all(check['pass'] for check in outer['checks'].values());count=count+1 if passed else 0
            outer.update(pass_this_sweep=passed,consecutive=count,converged=sweep>=c.outer_stop.min_sweeps and count>=c.outer_stop.consecutive_passes)
            row['outer_convergence']=outer
            atomic_json(folder/f'sweep_{sweep:04d}.json',row)
            np.savez_compressed(folder/'pending_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),H_used=np.asarray(H_used),
                damage_image=np.asarray(image),d_parent=np.asarray(d_parent),H_parent=np.asarray(H_parent))
            eqx.tree_serialise_leaves(folder/'pending_network.eqx',net)
            atomic_json(folder/'pending.json',dict(index=index,load=float(load),sweep=sweep,reaction=reaction,
                fields_sha256=sha(folder/'pending_fields.npz'),network_sha256=sha(folder/'pending_network.eqx'),certified_path=False))
            print(json.dumps(dict(event='outer',index=index,sweep=sweep,reaction=reaction,AT2=physical['AT2_relative'],crack=row['crack'],strict_ready=outer['converged'])),flush=True)
            if outer['converged']:break
        force=m.matrix(np.asarray(d))@np.asarray(q);host_norm=float(np.linalg.norm(force[m.free]))
        if host_norm>mechanical['convergence_limit']*(1+1e-3):raise RuntimeError('provisional host force equilibrium fails')
        np.testing.assert_allclose(force@m.load_direction,reaction,rtol=1e-9,atol=1e-6)
        np.testing.assert_array_equal(np.asarray(H),np.maximum(np.asarray(H_parent),m.driver(np.asarray(q))))
        shutil.copyfile(folder/'pending_fields.npz',folder/'fields.npz');shutil.copyfile(folder/'pending_network.eqx',folder/'network.eqx')
        report=dict(index=index,load=float(load),reaction=reaction,sweeps=sweep,seconds=time.perf_counter()-started,
            physical_AT2=physical['AT2_relative'],inner_AT2=inner['AT2_relative'],outer_convergence=outer,
            crack=row['crack'],healing=row['healing'],history_decrease=row['history_decrease'],host_force_norm=host_norm,
            **advancement(user_authorized=a.user_authorized,strict_ready=bool(outer['converged']),numerically_valid=True,provisional_ancestry=True),
            parent_sha256=parent_sha,state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'))
        atomic_json(folder/'result.json',report);curve.append(report);parent_sha=report['state_sha256'];publish(False,'running')
        print(json.dumps(dict(event='load_advanced',**report)),flush=True)
    if code!={name:sha(Path(__file__).with_name(name)) for name in names}:raise RuntimeError('provisional numerical source changed during run')
    publish(True,'path_completed_provisionally');print(json.dumps(dict(event='completed',last_index=index,reaction=reaction)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config']())
