"""Frozen learned velocity: GMLS Laplacian versus mapped SBP viscous flux."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def mapped_flux_divergence(fx,fy,data,*,xp):
    """Physical div(F) with compatible Y_x=D_xY and h=D_etaY metrics."""
    nx=len(data['transport_jacobian']);ny=len(data['transport_weight_eta'])
    if fx.shape!=fy.shape or fx.ndim!=3 or fx.shape[1]!=nx*ny:
        raise ValueError('Mapped fluxes require matching(time,ordered nodes,components) arrays')
    y=data['points'][:,1].reshape(nx,ny)
    y_x=xp.einsum('ij,jk->ik',data['transport_dx'],y)
    X=fx.reshape(-1,nx,ny,fx.shape[-1]);Y=fy.reshape(X.shape)
    h=data['transport_jacobian'][None,:,None,None]
    a=xp.einsum('ij,tjkc->tikc',data['transport_dx'],h*X)
    b=xp.einsum('ij,tkjc->tkic',data['transport_deta'],Y-y_x[None,:,:,None]*X)
    return ((a+b)/h).reshape(fx.shape)


def probe(run,output):
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,fields,spatial_field_jets
    from examples.channel_nsch2d.problem import quadrature
    from examples.channel_nsch2d.transport import build_transport
    from docs.tasks.channel_trained_operator_probe import local_laplacian
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
    run=Path(run);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'probe_source.py')
    def record(stage):
        (output/'progress.json').write_text(json.dumps(dict(stage=stage))+'\n');print(stage,flush=True)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if jax.default_backend()!='cpu':raise RuntimeError('Viscous consistency probe must remain CPU-only')
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key));e=encode(model,data)
    times=jnp.asarray([.025,.5,1.,6.],dtype=c.train.compute_dtype)
    @eqx.filter_jit
    def jets(net,encoded,points):return spatial_field_jets(net,encoded,points,times,data['amplitudes'],c)
    def evaluate(points):
        rows=[];chunk=2048
        for start in range(0,len(points),chunk):
            x=jnp.asarray(points[start:start+chunk],dtype=c.train.compute_dtype);count=len(x)
            x=jnp.concatenate((x,jnp.repeat(x[-1:],chunk-count,axis=0)))
            rows.append([np.asarray(a,dtype=np.float64)[:,:count] for a in jets(model,e,x)])
        return [np.concatenate([r[i] for r in rows],axis=1) for i in range(3)]
    ix=np.unique(np.rint(np.linspace(1,c.sample.Nx-1,32)).astype(int))
    iy=np.unique(np.rint(np.linspace(1,c.sample.Ny-1,32)).astype(int))
    ii,jj=np.meshgrid(ix,iy,indexing='ij');ids=(ii*(c.sample.Ny+1)+jj).ravel()
    if len(ids)!=1024:raise ValueError('Expected the declared1024physical comparison queries')
    @eqx.filter_jit
    def exact(net,encoded,x):return local_laplacian(lambda q:fields(net,encoded,q,times,data['amplitudes'],c)[...,2:4],x)
    record('exact_velocity_Laplacian')
    primal,lap=[np.asarray(a,dtype=np.float64) for a in exact(model,e,data['points'][ids])]
    record('original_grid_viscous_flux')
    original=evaluate(raw['points']);replay=0.
    for n,t in enumerate(np.asarray(times)):
        match=np.flatnonzero(np.isclose(saved['time'],t,atol=2e-7,rtol=0))
        if len(match):replay=max(replay,float(np.max(np.abs(original[0][n]-saved['fields'][match[0]]))))
    def divergence(values,payload):
        _,dx,dy=values;nu=c.physical.viscosity
        shear=nu*(dy[...,2]+dx[...,3])
        return mapped_flux_divergence(np.stack((2*nu*dx[...,2],shear),axis=-1),
            np.stack((shear,2*nu*dy[...,3]),axis=-1),payload,xp=np)
    coarse=divergence(original,raw)[:,ids]
    gmls=c.physical.viscosity*np.sum(raw['laplacian'][None,ids,:,None]*
        (original[0][:,raw['neighbors'][ids],2:4]-original[0][:,ids,None,2:4]),axis=2)
    fine=ConfigDict(c.to_dict());fine.sample.Nx*=2;fine.sample.Ny*=2
    points,_=quadrature(raw['amplitudes'],fine,fine.sample.Nx,fine.sample.Ny)
    refined_data=dict(points=points,**build_transport(points,raw['amplitudes'],fine))
    refined_ids=((2*ii)*(fine.sample.Ny+1)+2*jj).ravel()
    np.testing.assert_array_equal(points[refined_ids].astype(np.float32),raw['points'][ids].astype(np.float32))
    record('refined_grid_viscous_flux')
    refined=evaluate(points);div_fine=divergence(refined,refined_data)[:,refined_ids]
    primal_gap=max(float(np.max(np.abs(original[0][:,ids,2:4]-primal))),
        float(np.max(np.abs(refined[0][:,refined_ids,2:4]-primal))))
    if max(primal_gap,replay)>2e-5:raise ValueError(f'Frozen physical/replay gate failed:{primal_gap},{replay}')
    w=(raw['transport_measure']*raw['interior'])[ids];w/=w.sum()
    rms=lambda a:float(np.sqrt(np.sum(a*a*w[:,None])))
    rows=[]
    for n,t in enumerate(np.asarray(times)):
        expected=c.physical.viscosity*lap[n];norm=rms(expected)
        if norm<=0:raise ValueError('A nonzero AD viscous norm is required')
        rows.append(dict(time=float(t),viscous_AD_rms=norm,
            GMLS_relative=rms(gmls[n]-expected)/norm,
            SBP_flux_relative=rms(coarse[n]-expected)/norm,
            refined_SBP_flux_relative=rms(div_fine[n]-expected)/norm))
    result=dict(status='frozen_viscous_flux_consistency_completed',run=str(run),records=rows,
        saved_GPU_replay_gap=replay,primal_gap=primal_gap,queries=len(ids),
        scope='Frozen-network operator consistency; no new training, reference trajectory, physical BC or extra learned stress',
        summation='Host FP64 operators with CPU neural fields; not a GPU reduction emulation')
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');record('comparison_completed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.output)
