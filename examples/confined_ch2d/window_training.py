"""Sequential PDE-only PINN increments with fixed-shape JAX states/factors."""
from pathlib import Path
import json,shutil,time
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
from nonlocal_pinn.training.optimizers import make_optimizer
from examples.confined_ch2d.problem import load_reference,initial_field,relative,identity
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.preconditioning import BiharmonicMetric
from examples.confined_ch2d.window_network import build_window_network,window_fields,make_window_loss


def window_bounds(c):
    w=c.train.windows
    bounds=w.time_scale*np.expm1(np.linspace(0,1,w.count+1)*np.log1p(c.physical.terminal_time/w.time_scale))
    bounds[0]=0.;bounds[-1]=c.physical.terminal_time
    if not np.all(np.diff(bounds)>0):raise ValueError('Window boundaries must increase')
    return bounds


def field_metrics(phi,mu,ref):
    record={}
    for key,values in [('phi',phi),('mu',mu)]:
        for target in ('local','nonlocal'):
            truth=ref[key+'_'+target]
            record['pooled_rel_l2_'+key+'_'+target]=relative(values,truth,ref['volumes'])
            record['terminal_rel_l2_'+key+'_'+target]=relative(values[-1],truth[-1],ref['volumes'])
            record['per_time_rel_l2_'+key+'_'+target]=[relative(a,b,ref['volumes']) for a,b in zip(values,truth,strict=True)]
    return record


def train_windows(c):
    ref=load_reference(c);x=jnp.asarray(ref['points']);vol=jnp.asarray(ref['volumes'])
    initial=initial_field(x,c,xp=jnp);mean=jnp.dot(vol,initial)/vol.sum()
    D=RegionalLaplacian(ref['points'],ref['volumes'],c);bounds=window_bounds(c)
    kn,kt=jax.random.split(jax.random.PRNGKey(c.train.seed));net=build_window_network(c,kn)
    loss=make_window_loss(x,vol,D,mean,c);opt=make_optimizer(c)
    @eqx.filter_jit
    def step(model,state,start,width,times,metric):
        value,g=eqx.filter_value_and_grad(loss)(model,start,width,times,metric)
        updates,state=opt.update(g,state,model)
        return eqx.apply_updates(model,updates),state,value
    params,static=eqx.partition(net,eqx.is_inexact_array);_,unravel=ravel_pytree(params)
    @jax.jit
    def vector_value_grad(vector,start,width,times,metric):
        def objective(z):return loss(eqx.combine(unravel(z),static),start,width,times,metric)
        return jax.value_and_grad(objective)(vector)
    @eqx.filter_jit
    def endpoint(model,start,width):return window_fields(model,x,width,width,start,mean,vol,D,c)
    @eqx.filter_jit
    def queries(model,start,width,times):return jax.vmap(lambda t:window_fields(model,x,t,width,start,mean,vol,D,c))(times)
    run=Path(c.log.log_dir)/c.log.run_name
    if not c.log.run_name:raise ValueError('Windowed runs require an explicit unique run_name')
    run.mkdir(parents=True,exist_ok=False);(run/'config.json').write_text(c.to_json(indent=2)+'\n')
    root=Path(__file__).resolve().parents[2]
    shutil.copytree(Path(__file__).parent,run/'source/examples/confined_ch2d',ignore=shutil.ignore_patterns('__pycache__'))
    for rel in ('src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py','src/nonlocal_pinn/nn/arch.py','src/nonlocal_pinn/nn/embedding.py'):
        target=run/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
    manifest=dict(model='sequential_nonlocal_ch_pinn_v1',status='incomplete',identity=identity(c),bounds=bounds.tolist(),windows=[],training_targets='PDE only; preceding PINN endpoint, never reference state')
    prediction=np.full((len(ref['time']),len(x),2),np.nan)
    owner=np.minimum(np.searchsorted(bounds[1:],ref['time'],side='left'),len(bounds)-2)
    rows=[];samples=[];timing=[];start_clock=time.perf_counter();soap_done=lbfgs_done=0
    settings=c.train.windows
    with (run/'loss.jsonl').open('x') as log:
        for i,(left,right) in enumerate(zip(bounds[:-1],bounds[1:],strict=True)):
            width=jnp.asarray(right-left,dtype=jnp.float64);stage_clock=time.perf_counter()
            metric=BiharmonicMetric(D,ref['volumes'],time_scale=float(width),mobility=c.physical.mobility,epsilon=c.physical.epsilon)
            state=opt.init(eqx.filter(net,eqx.is_array))
            steps=settings.first_soap if i==0 else settings.soap_each
            for local_step in range(1,steps+1):
                kt,key=jax.random.split(kt)
                u=jax.random.uniform(key,(c.sample.Nt,),dtype=jnp.float64,minval=1e-6,maxval=1-1e-6)
                times=(jnp.arange(c.sample.Nt)+u)*width/c.sample.Nt
                samples.append(left+np.asarray(times))
                tick=time.perf_counter();net,state,value=step(net,state,initial,width,times,metric)
                value.block_until_ready();timing.append(time.perf_counter()-tick);soap_done+=1
                if not np.isfinite(float(value)):raise FloatingPointError('Nonfinite window SOAP loss')
                if local_step==1 or local_step%100==0:
                    log.write(json.dumps(dict(window=i,soap_step=soap_done,loss=float(value)),allow_nan=False)+'\n');log.flush()
            vector,_=ravel_pytree(eqx.filter(net,eqx.is_inexact_array))
            fixed=(jnp.arange(c.train.refinement.time_points,dtype=jnp.float64)+.5)*width/c.train.refinement.time_points
            limit=settings.first_lbfgs if i==0 else settings.lbfgs_each
            maxfun=settings.first_maxfun if i==0 else settings.maxfun_each
            def objective(z):
                value,g=vector_value_grad(jnp.asarray(z),initial,width,fixed,metric)
                a,b=float(value),np.asarray(g,dtype=np.float64)
                if not np.isfinite(a) or not np.isfinite(b).all():raise FloatingPointError('Nonfinite window L-BFGS objective')
                return a,b
            result=minimize(objective,np.asarray(vector),jac=True,method='L-BFGS-B',options=dict(maxiter=limit,maxfun=maxfun,
                maxls=c.train.refinement.max_line_search,maxcor=c.train.refinement.history_size,
                gtol=c.train.refinement.gradient_tolerance,ftol=c.train.refinement.function_tolerance))
            net=eqx.combine(unravel(jnp.asarray(result.x)),static);lbfgs_done+=int(result.nit)
            directory=run/f'window_{i:03d}';directory.mkdir()
            eqx.tree_serialise_leaves(directory/'model.eqx',net)
            np.savez_compressed(directory/'initial.npz',phi=np.asarray(initial),time=left)
            manifest['windows'].append(dict(model=f'window_{i:03d}/model.eqx',initial=f'window_{i:03d}/initial.npz',start=float(left),end=float(right)))
            end=endpoint(net,initial,width);mask=owner==i
            if np.any(mask):
                local=jnp.asarray(np.clip(ref['time']-left,0,float(width)))
                values=np.asarray(queries(net,initial,width,local));prediction[mask]=values[mask]
            available=owner<=i
            record=dict(window=i,start=float(left),end=float(right),soap_steps=steps,lbfgs_steps=int(result.nit),
                lbfgs_status=int(result.status),lbfgs_message=str(result.message),function_evaluations=int(result.nfev),
                final_loss=float(result.fun),elapsed_seconds=time.perf_counter()-start_clock,window_seconds=time.perf_counter()-stage_clock,
                available_reference_times=ref['time'][available].tolist(),
                prefix_phi_local=relative(prediction[available,:,0],ref['phi_local'][available],ref['volumes']))
            rows.append(record)
            with (run/'metrics.jsonl').open('a') as file:file.write(json.dumps(record,allow_nan=False)+'\n')
            print(f'[window {i+1}/{settings.count}] t={right:.6g} loss={result.fun:.5g} prefix_phiL2={record["prefix_phi_local"]:.5g}',flush=True)
            initial=end[:,0]
            (run/'trajectory_partial.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    if not np.isfinite(prediction).all():raise RuntimeError('Window trajectory did not cover reference query times')
    manifest['status']='complete';(run/'trajectory.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(run/'prediction.npz',time=ref['time'],points=ref['points'],volumes=ref['volumes'],phi=prediction[...,0],mu=prediction[...,1])
    np.savez_compressed(run/'training_times.npz',times=np.stack(samples),bounds=bounds)
    metrics=field_metrics(prediction[...,0],prediction[...,1],ref)
    summary=dict(cold_start=True,method='nonlocal_windows',wall_seconds=time.perf_counter()-start_clock,
        completed_soap=soap_done,completed_lbfgs=lbfgs_done,requested_soap=c.train.iterations,requested_lbfgs=c.train.refinement.iterations,
        mean_update_seconds_after_100=float(np.mean(timing[100:])),final_metrics=metrics,
        final_checkpoint='trajectory.json',best_selection='No reference selection; propagate each final PDE-only endpoint',windows=rows)
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2),flush=True)
    return run
