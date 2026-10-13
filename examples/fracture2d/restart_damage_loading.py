"""Certified52 neural restart with128 outer updates and original signed physics."""
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
from examples.fracture2d.restart_damage_source import restore_certified_source


def run(c):
    a=c.core.restart
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform!='gpu' or a.user_authorized is not True
            or a.initial_index!=52 or list(a.indices)!=list(range(53,76)) or c.stagger.omega!=.5
            or any(type(x) is not int or x<1 for x in [a.outer_updates,a.inner_updates])
            or a.outer_updates>c.outer_stop.max_sweeps or list(a.range_penalties)!=list(c.loading.penalties)
            or a.drift_envelope!=1.08 or a.outer_updates!=128 or a.inner_updates!=512
            or list(a.checkpoint_sweeps)!=[8,32,64,128]):
        raise ValueError('provisional branch requires explicit user authorization and original fixed-shape CUDA model')
    p=prepare_trial(c);original=p.mechanics
    m=AsymmetricLocalMechanics(original.coordinates,original.triangles,original.n_material,original.n_material_cells,
        original.volumes,E=c.physical.E,nu=c.physical.nu,k_res=c.physical.k_res,
        constrained=original.constrained,load_direction=original.load_direction,
        negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
    source_fields,net,source_report,pins=restore_certified_source(c,p,m)
    q,d,H=[source_fields[key] for key in ['q','d','H']]
    saved_image,saved_H=source_fields['damage_image'],source_fields['H_used']
    reaction=float(source_report['reaction'])
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
    source_load=jnp.asarray(a.initial_index*c.data.load_step,jnp.float64)
    old_context=context(saved_H,source_load,zeros,zeros,a.range_penalties[0])
    np.testing.assert_allclose(predict(net,old_context,source_load),saved_image,rtol=1e-10,atol=1e-12)
    current_context=context(H,source_load,zeros,zeros,a.range_penalties[0])
    np.testing.assert_allclose(predict(net,current_context,source_load),source_fields['raw_current_image'],rtol=1e-10,atol=1e-12)
    parity_context=context(p.H,p.load,zeros,zeros,a.range_penalties[0])
    v0,g0=eqx.filter_value_and_grad(objective)(net,parity_context,p.load)
    v1,g1=eqx.filter_value_and_grad(p.cached_objective)(net,p.full_context,p.load)
    np.testing.assert_allclose(v0,v1,rtol=1e-10,atol=1e-12)
    for x,y in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)):np.testing.assert_allclose(x,y,rtol=1e-8,atol=1e-9)
    fit=make_asymmetric_lbfgs(net,objective,metrics)[0]
    solve=m.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,maxiter=c.validation.cg_maxiter)
    graph=mesh_adjacency(m.triangles[:m.n_material_cells],population_size=m.n_material)
    def extent(damage):return crack_extent(damage,p.points,graph,threshold=a.crack_threshold,tip=a.tip,
        tip_radius=a.tip_radius,right_edge=a.right_edge,tolerance=a.coordinate_tolerance)
    def range_pass(values):return values['damage_min']>=-c.reporting.negative_range_atol and values['damage_max']<=1+c.reporting.positive_range_atol
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=False)
    names=['restart_damage_loading.py','restart_damage_source.py','provisional_damage.py','core_halo_damage_trial.py','core_halo_damage.py',
        'asymmetric_damage_range.py','local_mechanics.py','outer_stopping.py','range_damage_optimization.py']
    code={name:sha(Path(__file__).with_name(name)) for name in names}
    atomic_json(root/'identity.json',dict(config=c.to_dict(),geometry=p.geometry,code=code,source_pins=pins,
        user_authorized_provisional_advancement=True,original_strict_branch_modified=False,
        accepted_strict_prefix=52,restart_index=52,source_has_provisional_ancestry=False,
        initial_certified_path=True,direct_training_targets=False,
        damage_output_clipped=False,physics_unchanged=True,dynamic_full_H_feature_cache=True))
    (root/'code_snapshot').mkdir()
    for name in names:shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    shutil.copyfile(a.source_report,root/'source_certified_report.json')
    shutil.copyfile(a.source_audit,root/'source_audit.json')
    shutil.copyfile(a.source_parameterization,root/'parameterization.npz')
    shutil.copyfile(a.source_fields,root/'initial_fields.npz')
    shutil.copyfile(a.source_network,root/'initial_network.eqx')
    folder=root/f'state_{a.initial_index:04d}';folder.mkdir()
    shutil.copyfile(a.source_fields,folder/'fields.npz');shutil.copyfile(a.source_network,folder/'network.eqx')
    initial=dict(index=52,load=float(source_load),reaction=reaction,source_case=source_report,
        **advancement(user_authorized=True,strict_ready=True,numerically_valid=True,provisional_ancestry=False),
        crack=extent(d),state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'))
    atomic_json(folder/'result.json',initial);curve=[initial];parent_sha=initial['state_sha256'];provisional_ancestry=False
    q,d,H=jnp.asarray(q),jnp.asarray(d),jnp.asarray(H)
    def publish(complete,status):atomic_json(root/'result.json',dict(complete=complete,status=status,curve=curve,
        requested_indices=list(a.indices),all_states_locally_strict=all(row['local_strict_ready'] for row in curve),
        certified_path=all(row['certified_path'] for row in curve),user_provisional_ancestry=provisional_ancestry))
    publish(False,'running');print(json.dumps(dict(event='certified_source_restored',index=52,reaction=reaction)),flush=True)
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
                inner_stages=stages,crack=extent(np.asarray(d)),user_provisional_ancestry=provisional_ancestry)
            outer=evaluate_asymmetric(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count,
                negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
            outer['checks']['raw_current_H_AT2']=dict(value=raw['AT2_relative'],limit=c.reporting.coupled_AT2_relative_limit,
                **{'pass':raw['AT2_relative']<=c.reporting.coupled_AT2_relative_limit})
            outer['checks']['raw_current_H_range']=dict(value=max(0.,-raw['damage_min']),
                limit=c.reporting.negative_range_atol,**{'pass':range_pass(raw)})
            passed=all(check['pass'] for check in outer['checks'].values());count=count+1 if passed else 0
            outer.update(pass_this_sweep=passed,consecutive=count,converged=sweep>=c.outer_stop.min_sweeps and count>=c.outer_stop.consecutive_passes,
                reason='converged' if sweep>=c.outer_stop.min_sweeps and count>=c.outer_stop.consecutive_passes else 'iterating')
            row['outer_convergence']=outer
            atomic_json(folder/f'sweep_{sweep:04d}.json',row)
            np.savez_compressed(folder/'pending_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),H_used=np.asarray(H_used),
                damage_image=np.asarray(image),d_parent=np.asarray(d_parent),H_parent=np.asarray(H_parent))
            eqx.tree_serialise_leaves(folder/'pending_network.eqx',net)
            atomic_json(folder/'pending.json',dict(index=index,load=float(load),sweep=sweep,reaction=reaction,
                fields_sha256=sha(folder/'pending_fields.npz'),network_sha256=sha(folder/'pending_network.eqx'),certified_path=False))
            if sweep in a.checkpoint_sweeps or outer['converged']:
                checkpoint=folder/f'checkpoint_{sweep:04d}';checkpoint.mkdir()
                shutil.copyfile(folder/'pending_fields.npz',checkpoint/'fields.npz')
                shutil.copyfile(folder/'pending_network.eqx',checkpoint/'network.eqx')
                atomic_json(checkpoint/'report.json',dict(**row,fields_sha256=sha(checkpoint/'fields.npz'),
                    network_sha256=sha(checkpoint/'network.eqx'),physical_state_promoted=False))
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
            **advancement(user_authorized=a.user_authorized,strict_ready=bool(outer['converged']),numerically_valid=True,provisional_ancestry=provisional_ancestry),
            parent_sha256=parent_sha,state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'))
        atomic_json(folder/'result.json',report);curve.append(report);parent_sha=report['state_sha256']
        provisional_ancestry=provisional_ancestry or not report['local_strict_ready'];publish(False,'running')
        print(json.dumps(dict(event='load_advanced',**report)),flush=True)
    if code!={name:sha(Path(__file__).with_name(name)) for name in names}:raise RuntimeError('provisional numerical source changed during run')
    publish(True,'path_completed_provisionally' if provisional_ancestry else 'path_completed_certified');print(json.dumps(dict(event='completed',last_index=index,reaction=reaction)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config']())
