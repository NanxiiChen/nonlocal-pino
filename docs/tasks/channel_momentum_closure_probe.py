"""Frozen CPU global momentum closure; no training or evolved reference labels."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def global_momentum(value,rate,dx,dy,bv,bx,by,measure,boundary,bw,rho,nu,*,xp):
    """Area-normalized integrated NS residual with the ORIGINAL -phi*grad(mu)."""
    n=boundary[:,2:4];u=bv[...,2:4]
    shear=nu*(by[...,2]+bx[...,3]);p=bv[...,4]
    traction=xp.stack(((-p+2*nu*bx[...,2])*n[None,:,0]+shear*n[None,:,1],
        shear*n[None,:,0]+(-p+2*nu*by[...,3])*n[None,:,1]),axis=-1)
    outward=xp.sum(u*n[None,:,:],axis=-1)
    flux=rho*u*outward[...,None]-traction
    volume=rho*rate[...,2:4]+value[...,0,None]*xp.stack((dx[...,1],dy[...,1]),axis=-1)
    return (xp.sum(volume*measure[None,:,None],axis=1)+xp.sum(flux*bw[None,:,None],axis=1))/xp.sum(measure)


def probe(run,output):
    # This probe is deliberately CPU-only while the independent GPU trial runs.
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,fields,spatial_field_jets
    from examples.channel_nsch2d.problem import boundary,quadrature
    from examples.channel_nsch2d.transport import build_transport
    from examples.channel_nsch2d.temporal import previous_times,backward_rate
    run=Path(run);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'probe_source.py')
    def record(stage,**extra):
        (output/'progress.json').write_text(json.dumps(dict(stage=stage,**extra),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    if hasattr(os,'sched_getaffinity'):
        cores=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cores[-8:])
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if 'velocity' not in c or 'time_discretization' not in c or 'phase_transport' not in c:
        raise ValueError('This diagnostic requires the declared incompressible SBP/BDF2 model')
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if jax.default_backend()!='cpu':raise RuntimeError('Momentum diagnosis must remain CPU-only')
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key));encoded=encode(net,data)
    times=np.asarray([.025,.5,1.,6.],dtype=c.train.compute_dtype)
    @eqx.filter_jit
    def jets(net,e,x,t):
        value,dx,dy=spatial_field_jets(net,e,x,t,data['amplitudes'],c)
        t1,t2=previous_times(t,c.time_discretization.dt)
        history=fields(net,e,x,jnp.concatenate((t1,t2)),data['amplitudes'],c)
        rate=backward_rate(value,history[:len(t)],history[len(t):],t,c.time_discretization.dt)
        return value,rate,dx,dy
    def evaluate(points):
        rows=[];chunk=2048
        for start in range(0,len(points),chunk):
            x=jnp.asarray(points[start:start+chunk],dtype=c.train.compute_dtype);count=len(x)
            x=jnp.concatenate((x,jnp.repeat(x[-1:],chunk-count,axis=0)))
            rows.append([np.asarray(a,dtype=np.float64)[:,:count] for a in jets(net,encoded,x,jnp.asarray(times))])
        return [np.concatenate([row[i] for row in rows],axis=1) for i in range(4)]
    record('original_volume_and_boundary_jets')
    original=evaluate(raw['points']);boundaries={}
    for count in (48,96,192,384):
        selected=ConfigDict(c.to_dict());selected.sample.boundary_points=count
        b,w=boundary(raw['amplitudes'],selected);v,_,dx,dy=evaluate(b[:,:2])
        boundaries[count]=(v,dx,dy,b,w)
        record('boundary_quadrature_completed',points_per_side=count)
    fine=ConfigDict(c.to_dict());fine.sample.Nx*=2;fine.sample.Ny*=2
    points,_=quadrature(raw['amplitudes'],fine,fine.sample.Nx,fine.sample.Ny)
    measure=build_transport(points,raw['amplitudes'],fine)['transport_measure']
    record('refined_volume_jets')
    refined=evaluate(points);replay=0.
    finest_config=ConfigDict(c.to_dict());finest_config.sample.Nx*=4;finest_config.sample.Ny*=4
    finest_points,_=quadrature(raw['amplitudes'],finest_config,finest_config.sample.Nx,finest_config.sample.Ny)
    finest_measure=build_transport(finest_points,raw['amplitudes'],finest_config)['transport_measure']
    record('fourfold_volume_jets')
    fourfold=evaluate(finest_points)
    for i,t in enumerate(times):
        match=np.flatnonzero(np.isclose(saved['time'],t,atol=2e-7,rtol=0))
        if len(match):replay=max(replay,float(np.max(np.abs(original[0][i]-saved['fields'][match[0]]))))
    if replay>2e-5:raise ValueError(f'CPU saved-GPU physical replay failed:{replay}')
    def weak(values,w,b):
        bv,bx,by,points,bw=b
        return global_momentum(*values,bv,bx,by,w,points,bw,c.physical.density,c.physical.viscosity,xp=np)
    balances={f'boundary_{count}':weak(original,raw['transport_measure'],b) for count,b in boundaries.items()}
    balances['volume2_boundary384']=weak(refined,measure,boundaries[384])
    balances['volume4_boundary384']=weak(fourfold,finest_measure,boundaries[384])
    value,dt,dx,dy=original
    lap=np.sum(raw['laplacian'][None,:,:,None]*(value[:,raw['neighbors'],:]-value[:,:,None,:]),axis=2)
    strong=c.physical.density*(dt[...,2:4]+value[...,2,None]*dx[...,2:4]+value[...,3,None]*dy[...,2:4])
    strong+=np.stack((dx[...,4],dy[...,4]),axis=-1)-c.physical.viscosity*lap[...,2:4]
    strong+=value[...,0,None]*np.stack((dx[...,1],dy[...,1]),axis=-1)
    for name,mask in (('strong_GMLS_full_mean',np.ones_like(raw['interior'])),('strong_GMLS_interior_mean',raw['interior'])):
        w=raw['transport_measure']*mask
        balances[name]=np.sum(strong*w[None,:,None],axis=1)/w.sum()
    finest=balances['volume4_boundary384']
    bgap=np.linalg.norm(balances['boundary_384']-balances['boundary_192'],axis=-1)
    vgap=np.linalg.norm(balances['volume2_boundary384']-finest,axis=-1)
    # Explicit diagnostic agreement criterion, not a training denominator floor.
    tolerance=1e-4+.05*np.linalg.norm(finest,axis=-1)
    passed=all(np.isfinite(v).all() for v in balances.values()) and np.all(bgap<=tolerance) and np.all(vgap<=tolerance)
    result=dict(status='frozen_global_momentum_closure_completed' if passed else 'unvalidated_momentum_quadrature',run=str(run),backend=jax.default_backend(),
        time=times.tolist(),balances={k:v.tolist() for k,v in balances.items()},saved_GPU_replay_gap=replay,
        boundary_gap=bgap.tolist(),volume_gap=vgap.tolist(),quadrature_limits=tolerance.tolist(),
        original_to_double_volume_gap=np.linalg.norm(balances['boundary_384']-balances['volume2_boundary384'],axis=-1).tolist(),
        volume_grids=[[c.sample.Nx,c.sample.Ny],[fine.sample.Nx,fine.sample.Ny],[finest_config.sample.Nx,finest_config.sample.Ny]],
        context='Original learned model/encoder inputs and quadrature; only diagnostic query quadratures change',
        strong_GMLS_definition='Saved host stencil coefficients with CPU physical fields and host FP64 summation, not a GPU reduction emulation',
        scope='Constant-test momentum balance, not a trained loss, new solution/reference, unique pressure cause or L2 improvement')
    if not passed:
        (output/'comparison-unvalidated.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        failure=dict(status='quadrature_agreement_failed',boundary_gap=bgap.tolist(),volume_gap=vgap.tolist(),limits=tolerance.tolist())
        (output/'failed-gate.json').write_text(json.dumps(failure,indent=2,allow_nan=False)+'\n')
        record('diagnostic_gate_failed',saved_GPU_replay_gap=replay)
        raise ValueError(f'Weak diagnostic quadrature is not converged:{failure}')
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    record('diagnostic_completed',report=str(output/'comparison.json'),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.output)
