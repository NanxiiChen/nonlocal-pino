"""Physics-only auxiliary stress-head block solve with original fields fixed."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def statistics(x,xx,xy,momentum,target,wm,wc):
    """Quadratic of Rx=r0x-Xx*c1-Xy*c2, Ry=r0y+Xy*c1-Xx*c2."""
    import numpy as np
    if (x.ndim!=2 or xx.shape!=x.shape or xy.shape!=x.shape or momentum.shape!=(len(x),2)
            or target.shape!=momentum.shape or wm.shape!=(len(x),) or wc.shape!=wm.shape):
        raise ValueError('Stress block rows, targets and both measures must align')
    if (any(not np.isfinite(v).all() for v in (x,xx,xy,momentum,target,wm,wc))
            or np.any(wm<0) or np.any(wc<0) or wm.sum()<=0 or wc.sum()<=0):
        raise ValueError('Finite rows and nonnegative nonzero momentum/constitutive measures required')
    a=xx.T@(wm[:,None]*xx)+xy.T@(wm[:,None]*xy)+x.T@(wc[:,None]*x)
    b=xx.T@(wm[:,None]*xy)-xy.T@(wm[:,None]*xx)
    g1=-xx.T@(wm*momentum[:,0])+xy.T@(wm*momentum[:,1])-x.T@(wc*target[:,0])
    g2=-xy.T@(wm*momentum[:,0])-xx.T@(wm*momentum[:,1])-x.T@(wc*target[:,1])
    f=float(np.sum(wm[:,None]*momentum**2)+np.sum(wc[:,None]*target**2))
    return np.block([[a,b],[b.T,a]]),np.concatenate((g1,g2)),f


def fit(parent_run,stress_run,output):
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,fields,spatial_field_jets,extend_stress_outputs,stress_jets
    from examples.channel_nsch2d.temporal import previous_times,backward_rate
    from examples.channel_nsch2d.losses import residual_components
    from examples.channel_nsch2d.pressure_head import feature_jets
    from examples.channel_nsch2d.stress import constitutive,divergence
    parent_run=Path(parent_run);stress_run=Path(stress_run);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'fit_source.py')
    def record(stage,**extra):
        (output/'progress.json').write_text(json.dumps(dict(stage=stage,**extra),indent=2,allow_nan=False)+'\n')
        print(stage,extra,flush=True)
    c=ConfigDict(json.loads((parent_run/'config.json').read_text()));s=ConfigDict(json.loads((stress_run/'config.json').read_text()))
    if c.train.compute_dtype!='float32':raise ValueError('This rounded-head diagnostic requires the actual FP32 parent')
    if Path(s.train.refinement.parent_run)!=parent_run or c.operator.train_conditions!=1:
        raise ValueError('This diagnostic needs the actual single-geometry parent of the stress trial')
    if 'stress_formulation' in c or s.stress_formulation.initialization!='preserve_original':
        raise ValueError('Stress equilibration needs the original four-output parent and isotropic control')
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64');jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if jax.default_backend()!='cpu':raise RuntimeError('Frozen stress-block diagnostic must remain CPU-only')
    with np.load(parent_run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    with np.load(parent_run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));parent=eqx.tree_deserialise_leaves(parent_run/'model_final.eqx',build_network(c,key))
    e=encode(parent,data);n=len(raw['points']);points=np.concatenate((raw['points'],raw['boundary'][:,:2]))
    w=raw['transport_measure']*raw['interior'];w=w/w.sum()
    wb=raw['boundary_weights']/raw['boundary_weights'].sum()
    count=round(c.physical.terminal_time/c.time_discretization.dt);batch=8
    if count%batch:raise ValueError('Complete BDF support must divide the fixed8time diagnostic batch')
    nodes=np.arange(1,count+1,dtype=c.train.compute_dtype)*np.asarray(c.time_discretization.dt,dtype=c.train.compute_dtype)
    wm=np.concatenate((w,np.zeros(len(wb))))/count/c.loss.momentum_scale**2
    wc=np.concatenate((w,wb))/count/s.stress_formulation.constitutive_scale**2
    @eqx.filter_jit
    def original(model,encoded,x,t):
        value,dx,dy=spatial_field_jets(model,encoded,x,t,data['amplitudes'],c)
        a,b=previous_times(t,c.time_discretization.dt)
        history=fields(model,encoded,x,jnp.concatenate((a,b)),data['amplitudes'],c)
        rate=backward_rate(value,history[:len(t)],history[len(t):],t,c.time_discretization.dt)
        r0=residual_components(value,rate,dx,dy,jnp.zeros_like(value),c)[...,2:4]
        x0,xx,xy=feature_jets(model,encoded,x,t,c)
        target=c.physical.viscosity*jnp.stack((dx[...,2]-dy[...,3],dy[...,2]+dx[...,3]),axis=-1)
        return x0,xx,xy,r0,target,value
    @eqx.filter_jit
    def actual(model,encoded,x,t):
        value,dx,dy=spatial_field_jets(model,encoded,x,t,data['amplitudes'],s)
        a,b=previous_times(t,c.time_discretization.dt)
        history=fields(model,encoded,x,jnp.concatenate((a,b)),data['amplitudes'],s)
        rate=backward_rate(value,history[:len(t)],history[len(t):],t,c.time_discretization.dt)
        sigma,sx,sy=stress_jets(model,encoded,x,t,s)
        r=residual_components(value,rate,dx,dy,jnp.zeros_like(value),s,stress_divergence=divergence(sx,sy,xp=jnp))[...,2:4]
        residual=constitutive(sigma,dx,dy,c.physical.viscosity,xp=jnp)
        return r,residual,value
    chunk=2048;width=parent.width+1;G=np.zeros((2*width,2*width));g=np.zeros(2*width);constant=0.;replay=0.;basis_gap=0.
    pressure=np.concatenate((np.asarray(parent.output.weight[3]),np.asarray(parent.output.bias[3,None])))
    def batches():
        for offset in range(0,count,batch):
            t=nodes[offset:offset+batch]
            for start in range(0,len(points),chunk):
                q=points[start:start+chunk];size=len(q)
                q=np.concatenate((q,np.repeat(q[-1:],chunk-size,axis=0)))
                yield offset,start,size,t,jnp.asarray(q,dtype=c.train.compute_dtype)
    def check_replay(value,start,size,t):
        gap=0.;end=min(start+size,n)
        if start<n:
            for i,time in enumerate(t):
                match=np.flatnonzero(np.isclose(saved['time'],time,atol=2e-7,rtol=0))
                if len(match):gap=max(gap,float(np.max(np.abs(value[i,:end-start]-saved['fields'][match[0],start:end]))))
        return gap
    record('assembling_frozen_physics_stress_block',time_nodes=count)
    for offset,start,size,t,q in batches():
        vals=[np.asarray(a,dtype=np.float64)[:,:size] for a in original(parent,e,q,jnp.asarray(t))]
        x,xx,xy,r0,target,value=vals
        replay=max(replay,check_replay(value,start,size,t));basis_gap=max(basis_gap,float(np.max(np.abs(x@pressure-value[...,4]))))
        a,b,f=statistics(x.reshape(-1,width),xx.reshape(-1,width),xy.reshape(-1,width),r0.reshape(-1,2),target.reshape(-1,2),
            np.broadcast_to(wm[start:start+size],(len(t),size)).ravel(),np.broadcast_to(wc[start:start+size],(len(t),size)).ravel())
        G+=a;g+=b;constant+=f
        if start==0:record('assembling_frozen_physics_stress_block',completed_time_nodes=offset+len(t),time_nodes=count)
    first=json.loads((stress_run/'refinement_evaluations.jsonl').read_text().splitlines()[0])['parts']
    expected=sum(first[k] for k in ('momentum_x','momentum_y','stress_normal','stress_shear','boundary_stress_normal','boundary_stress_shear'))
    if max(replay,basis_gap)>2e-5 or abs(constant-expected)>.01*expected:
        raise ValueError(f'Frozen stress-block replay/objective mismatch:{replay},{basis_gap},{constant},{expected}')
    G=(G+G.T)/2;diag=np.diag(G)
    if not np.isfinite(G).all() or np.any(diag<=0):raise ValueError('Finite positive Gram diagonals required; no rank fallback')
    scale=np.sqrt(diag);H=G/scale[:,None]/scale[None,:];linear=g/scale
    eigen=np.linalg.eigvalsh(H)
    if eigen[0]<-1e-10*eigen[-1]:raise ValueError('Stress physics quadratic is not positive semidefinite')
    record('solving_physics_head_block',objective_before=constant)
    objective=lambda b:(constant+2*linear@b+b@H@b,2*(linear+H@b))
    result=minimize(objective,np.zeros(2*width),method='L-BFGS-B',jac=True,
        options=dict(maxiter=500,maxfun=2000,maxls=50,maxcor=20,gtol=1e-10,ftol=1e-14))
    coef=(result.x/scale).reshape(2,width)
    if not np.isfinite(coef).all() or not result.fun<constant:raise ValueError('No finite stress-block improvement')
    model=extend_stress_outputs(parent,build_network(s,key));dtype=model.output.weight.dtype
    d1=jnp.asarray(coef[0],dtype=dtype);d2=jnp.asarray(coef[1],dtype=dtype)
    weights=model.output.weight.at[3].add(d1[:-1]).at[5].add(-d1[:-1]).at[4].set(d2[:-1])
    bias=model.output.bias.at[3].add(d1[-1]).at[5].add(-d1[-1]).at[4].set(d2[-1])
    model=eqx.tree_at(lambda m:(m.output.weight,m.output.bias),model,(weights,bias))
    value_after=0.;replay_after=0.
    record('verifying_actual_rounded_stress_block')
    for offset,start,size,t,q in batches():
        r,cr,value=[np.asarray(a,dtype=np.float64)[:,:size] for a in actual(model,e,q,jnp.asarray(t))]
        value_after+=float(np.sum(r*r*wm[None,start:start+size,None])+np.sum(cr*cr*wc[None,start:start+size,None]))
        replay_after=max(replay_after,check_replay(value,start,size,t))
    if (not np.isfinite(value_after) or value_after>=constant or replay_after>2e-5
            or abs(value_after-result.fun)>1e-6+.001*abs(result.fun)):
        raise ValueError(f'Actual rounded stress-block reduction/replay failed:{value_after},{result.fun},{replay_after}')
    eqx.tree_serialise_leaves(output/'model_projected.eqx',model)
    np.savez_compressed(output/'quadratic.npz',gram=G,linear=g,constant=constant,coefficients=coef,time=nodes)
    report=dict(status='physics_stress_head_equilibration_passed',parent_run=str(parent_run),stress_control=str(stress_run),
        parent_config=c.to_dict(),stress_formulation=s.stress_formulation.to_dict(),time_nodes=count,
        input_nodes=n,objective_before=constant,actual_GPU_initial_relevant_objective=expected,
        objective_after_quadratic=float(result.fun),objective_after_actual_FP32=value_after,
        old_GPU_physical_replay_gap=replay,projected_GPU_physical_replay_gap=replay_after,
        pressure_feature_gap=basis_gap,scaled_gram_eigenvalue_range=[float(eigen[0]),float(eigen[-1])],
        optimizer=dict(iterations=int(result.nit),status=int(result.status),success=bool(result.success),message=str(result.message)),
        coefficient_norm=float(np.linalg.norm(coef)),coefficient_max=float(np.max(np.abs(coef))),
        projected_model=str(output/'model_projected.eqx'),supervised_evolved_field_samples=0,
        scope='Only auxiliary deviatoric/shear head block optimized; original physical fields fixed; no coupled-training L2 gain claim')
    (output/'fit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');record('equilibration_completed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run',required=True);p.add_argument('--stress-run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();fit(a.parent_run,a.stress_run,a.output)
