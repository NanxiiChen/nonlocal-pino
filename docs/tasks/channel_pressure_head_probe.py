"""Finite physics-only pressure-head diagnostic, after a failed accuracy gate."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def probe(after_pid,progress_path,run,output,chemical_shift,physical_baseline_fit):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'probe_source.py')
    source=Path(__file__).resolve().parents[2]/'examples/channel_nsch2d/pressure_head.py'
    shutil.copyfile(source,output/'pressure_head_source.py')
    source=source.with_name('network.py');shutil.copyfile(source,output/'network_source.py')
    def record(stage,**extra):
        report=dict(stage=stage,**extra)
        (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report,allow_nan=False),flush=True)
    record('waiting_for_time_refinement',pid=after_pid)
    while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    previous=json.loads(Path(progress_path).read_text())
    if previous['stage']=='online_training_and_input_replay_completed_unseen_validation_required':
        record('skipped_because_online_recipe_proceeded',goal_complete=False);return
    if previous['stage']!='single_geometry_target_still_unmet':
        raise ValueError('Pressure diagnostic requires an actually completed failed accuracy gate')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError(f'Another GPU job remains active: {apps}')
    os.environ['JAX_PLATFORMS']='cuda'
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.online import validate_payload
    from examples.channel_nsch2d.losses import residual_components
    from examples.channel_nsch2d.network import build_network,encode,fields,spatial_field_jets
    from examples.channel_nsch2d.temporal import backward_rate,previous_times
    from examples.channel_nsch2d.pressure_head import features,feature_jets,quadratic_statistics

    run=Path(run);parent_config=ConfigDict(json.loads((run/'config.json').read_text()))
    c=ConfigDict(parent_config.to_dict())
    if 'pressure_coordinate' in c:raise ValueError('This comparison requires an original physical-pressure parent')
    if chemical_shift:
        if physical_baseline_fit is None:raise ValueError('Chemical-shift comparison requires its actual physical-pressure fit report')
        c.pressure_coordinate=ConfigDict(dict(kind='chemical_shift'))
    elif physical_baseline_fit is not None:
        raise ValueError('Physical baseline report belongs only to the explicit shifted-coordinate comparison')
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if c.operator.train_conditions!=1 or any(k in c for k in ('balance','stagger')):
        raise ValueError('Frozen pressure probe requires the uniform single-geometry objective')
    with np.load(run/'known_inputs.npz',allow_pickle=False) as saved_inputs:
        if any(a.shape[0]!=1 for a in saved_inputs.values()):
            raise ValueError('Pressure probe requires the actual one-geometry training payload')
        raw={k:saved_inputs[k][0] for k in saved_inputs.files}
    np.testing.assert_array_equal(raw['amplitudes'],[c.geometry.upper,c.geometry.lower])
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    validate_payload({k:np.asarray(v) for k,v in data.items()},c)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    encoded=encode(net,data)
    count=int(round(c.physical.terminal_time/c.time_discretization.dt))
    baseline=None
    if chemical_shift:
        baseline=json.loads(Path(physical_baseline_fit).read_text())
        if baseline['status']!='pressure_only_physics_fit_completed' or Path(baseline['run'])!=run or baseline['time_nodes']!=count:
            raise ValueError('Physical-pressure fit report does not match the frozen parent and time grid')
    if count%c.sample.Nt:raise ValueError('All BDF nodes must form complete fixed-size batches')
    node_times=np.arange(1,count+1,dtype=c.train.compute_dtype)*c.time_discretization.dt
    outlet=np.asarray(data['boundary'])[:,4]==2
    point_measure=raw['transport_measure'] if 'phase_transport' in c else raw['volumes']
    point_measure=point_measure*raw['interior'];point_measure=point_measure/point_measure.sum()/count
    wall_measure=raw['boundary_weights'][outlet];wall_measure=wall_measure/wall_measure.sum()/count

    @eqx.filter_jit
    def batch(model,condition,d,times):
        value,dx,dy=spatial_field_jets(model,condition,d['points'],times,d['amplitudes'],c)
        t1,t2=previous_times(times,c.time_discretization.dt)
        history=fields(model,condition,d['points'],jnp.concatenate((t1,t2)),d['amplitudes'],c)
        rate=backward_rate(value,history[:len(times)],history[len(times):],times,c.time_discretization.dt)
        lap=jnp.sum(d['laplacian'][None,:,:,None]*(value[:,d['neighbors'],:]-value[:,:,None,:]),axis=2)
        momentum=residual_components(value,rate,dx,dy,lap,c)[...,2:4]/c.loss.momentum_scale
        basis,bx,by=feature_jets(model,condition,d['points'],times,c)
        q=d['boundary'][outlet,:2]
        v,x,_=spatial_field_jets(model,condition,q,times,d['amplitudes'],c)
        b=features(model,condition,q,times,c)
        traction=(-v[...,4]+2*c.physical.viscosity*x[...,2])/c.loss.traction_scale
        coordinate=value[...,4]+value[...,0]*value[...,1] if chemical_shift else value[...,4]
        return (bx/c.loss.momentum_scale,by/c.loss.momentum_scale,momentum,
                -b/c.loss.traction_scale,traction,basis,coordinate)

    width=net.width+1;G=np.zeros((width,width));g=np.zeros(width);f=0.;maximum_gap=0.
    coefficient=np.concatenate((np.asarray(net.output.weight[3]),np.asarray(net.output.bias[3,None])))
    record('assembling_all_time_physics_quadratic',time_nodes=count,spatial_nodes=len(raw['points']))
    begin=time.perf_counter()
    for offset in range(0,count,c.sample.Nt):
        bx,by,r,b,t,basis,p=[np.asarray(a,dtype=np.float64) for a in batch(net,encoded,data,jnp.asarray(node_times[offset:offset+c.sample.Nt]))]
        maximum_gap=max(maximum_gap,float(np.max(np.abs(basis@coefficient-p))))
        for rows,error,measure in ((bx,r[...,0],point_measure),(by,r[...,1],point_measure),(b,t,wall_measure)):
            GG,gg,ff=quadratic_statistics(rows.reshape(-1,width),error.ravel(),np.broadcast_to(measure,error.shape).ravel())
            G+=GG;g+=gg;f+=ff
        record('assembling_all_time_physics_quadratic',completed_time_nodes=offset+c.sample.Nt,
               total_time_nodes=count,elapsed_seconds=time.perf_counter()-begin)
    if maximum_gap>2e-5:raise ValueError(f'Actual pressure-feature identity failed: {maximum_gap}')
    diagonal=np.diag(G)
    if not np.isfinite(G).all() or np.any(diagonal<=0):
        raise ValueError('Pressure quadratic requires finite, positive column response; no silent rank regularization')
    G=(G+G.T)/2
    scale=np.sqrt(diagonal);H=G/scale[:,None]/scale[None,:];b=g/scale
    eigen=np.linalg.eigvalsh(H)
    if eigen[0]<-1e-10*eigen[-1]:raise ValueError('Pressure Gram matrix is not positive semidefinite')
    np.savez_compressed(output/'quadratic.npz',gram=G,linear=g,constant=f,time=node_times,scale=scale)
    def objective(beta):return f+2*b@beta+beta@H@beta,2*(b+H@beta)
    result=minimize(objective,np.zeros(width),method='L-BFGS-B',jac=True,
        options=dict(maxiter=5000,maxfun=12000,maxls=50,maxcor=20,gtol=1e-10,ftol=1e-14))
    delta=result.x/scale
    if not np.isfinite(delta).all() or result.fun>=f:
        raise ValueError('No finite reduction of the original pressure-dependent physics objective')
    changed=eqx.tree_at(lambda n:(n.output.weight,n.output.bias),net,
        (net.output.weight.at[3].add(jnp.asarray(delta[:-1],dtype=net.output.weight.dtype)),
         net.output.bias.at[3].add(jnp.asarray(delta[-1],dtype=net.output.bias.dtype))))
    # Re-evaluate the actual rounded neural residuals; a smaller FP64 quadratic
    # alone is insufficient when a nearly dependent basis needs large weights.
    actual_objective=0.
    record('verifying_actual_rounded_physics_residuals')
    for offset in range(0,count,c.sample.Nt):
        values=batch(changed,encoded,data,jnp.asarray(node_times[offset:offset+c.sample.Nt]))
        momentum=np.asarray(values[2],dtype=np.float64);traction=np.asarray(values[4],dtype=np.float64)
        actual_objective+=float(np.sum(momentum**2*point_measure[None,:,None]))
        actual_objective+=float(np.sum(traction**2*wall_measure[None,:]))
    if not np.isfinite(actual_objective) or actual_objective>=f:
        raise ValueError('Rounded physical neural residuals do not improve; reject the pressure probe')
    eqx.tree_serialise_leaves(output/'pressure_head_model.eqx',changed)
    fit=dict(status='pressure_only_physics_fit_completed',run=str(run),time_nodes=count,
        pressure_coordinate='chemical_shift' if chemical_shift else 'physical',
        objective_before=f,objective_after_FP64_quadratic=float(result.fun),
        objective_after_actual_neural_residual=actual_objective,pressure_basis_max_gap=maximum_gap,
        optimizer_success=bool(result.success),optimizer_status=int(result.status),message=str(result.message),
        iterations=int(result.nit),function_evaluations=int(result.nfev),gradient_inf=float(np.max(np.abs(result.jac))),
        scaled_gram_eigenvalue_range=[float(eigen[0]),float(eigen[-1])],
        delta_norm=float(np.linalg.norm(delta)),delta_max=float(np.max(np.abs(delta))),
        fit_seconds=time.perf_counter()-begin,supervised_evolved_field_samples=0,
        compute_dtype=c.train.compute_dtype,statistics_dtype='float64_host',
        scope='Original momentum and outlet traction only; all BDF nodes/full grid; no regularization or reference-dependent fitting')
    if baseline is not None:
        fit.update(physical_parent_objective=baseline['objective_before'],
            optimized_physical_coordinate_objective=baseline['objective_after_actual_neural_residual'])
    (output/'fit.json').write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n')
    # Read evolved fields only after the physics-only optimization has ended.
    with np.load(run/'dense-fem-prediction.npz',allow_pickle=False) as a:saved={k:a[k] for k in a.files}
    evaluate=eqx.filter_jit(lambda model:fields(model,encoded,jnp.asarray(saved['points'],dtype=c.train.compute_dtype),
        jnp.asarray(saved['time'],dtype=c.train.compute_dtype),data['amplitudes'],c))
    if chemical_shift:
        original=eqx.filter_jit(lambda model:fields(model,encoded,jnp.asarray(saved['points'],dtype=c.train.compute_dtype),
            jnp.asarray(saved['time'],dtype=c.train.compute_dtype),data['amplitudes'],parent_config))
        before=np.asarray(original(net),dtype=np.float64)
    else:before=np.asarray(evaluate(net),dtype=np.float64)
    after=np.asarray(evaluate(changed),dtype=np.float64)
    gap=float(np.max(np.abs(before-saved['fields'])))
    other_gap=float(np.max(np.abs(before[...,:4]-after[...,:4])))
    if not np.isfinite(after).all() or max(gap,other_gap)>2e-5:
        raise ValueError(f'Frozen physical replay failed: parent={gap},other_fields={other_gap}')
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics={}
    for label,sl,start in (('phi',slice(0,1),0),('velocity',slice(2,4),1),('pressure',slice(4,5),1)):
        metrics[label]=dict(before=relative(before[start:,...,sl],saved['reference'][start:,...,sl]),
            after=relative(after[start:,...,sl],saved['reference'][start:,...,sl]))
    metrics['terminal_phi']=dict(before=relative(before[-1,:,0],saved['reference'][-1,:,0]),
        after=relative(after[-1,:,0],saved['reference'][-1,:,0]))
    np.savez_compressed(output/'prediction.npz',fields=after,points=saved['points'],time=saved['time'],
        amplitudes=saved['amplitudes'],reference=saved['reference'])
    record('frozen_pressure_head_diagnostic_completed',fit=fit,metrics=metrics,
        parent_replay_max_gap=gap,unchanged_physical_fields_max_gap=other_gap,
        principal_target_met=all(metrics[k]['after']<.1 for k in ('phi','terminal_phi','velocity','pressure')),
        canonical_training_promoted=False,goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--progress',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--chemical-shift',action='store_true');p.add_argument('--physical-baseline-fit')
    a=p.parse_args();probe(a.after_pid,a.progress,a.run,a.output,a.chemical_shift,a.physical_baseline_fit)
