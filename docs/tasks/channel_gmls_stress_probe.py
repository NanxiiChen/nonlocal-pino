"""Frozen exact viscous stresses, differentiated by polynomial GMLS gradients."""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def gradient_stencils(points,c):
    """QR functional reconstruction: Dx(x/r)=Dy(y/r)=1/r at each center."""
    import numpy as np
    from scipy.spatial import cKDTree
    x=np.asarray(points,dtype=np.float64);n=len(x);k=c.nl.neighbors;degree=c.nl.degree
    if c.nl.method!='gmls' or x.shape!=(n,2) or not np.isfinite(x).all():
        raise ValueError('Explicit GMLS and finite(N,2)physical nodes required')
    basis=[(a,b) for total in range(degree+1) for a in range(total+1) for b in [total-a]]
    if degree<1 or not len(basis)<k<=n or c.nl.weight_exponent<=0:
        raise ValueError('Positive GMLS degree/weights and sufficient neighbors required')
    distance,indices=cKDTree(x).query(x,k=k)
    if np.any(distance[:,1]<=0):raise ValueError('Duplicate physical nodes')
    radius=1.25*distance[:,-1];delta=(x[indices]-x[:,None,:])/radius[:,None,None]
    P=np.stack([delta[...,0]**a*delta[...,1]**b/math.factorial(a)/math.factorial(b) for a,b in basis],axis=-1)
    weight=(1-distance/radius[:,None])**c.nl.weight_exponent
    Q,R=np.linalg.qr(np.sqrt(weight)[...,None]*P,mode='reduced');condition=np.linalg.cond(R)
    if not np.isfinite(condition).all() or np.max(condition)>c.nl.condition_limit:
        raise ValueError(f'Unresolved gradient GMLS stencil:{np.max(condition)}')
    target=np.zeros((n,len(basis),2));target[:,basis.index((1,0)),0]=1/radius
    target[:,basis.index((0,1)),1]=1/radius
    coefficients=np.linalg.solve(np.swapaxes(R,-1,-2),target)
    stencil=np.sqrt(weight)[...,None]*np.einsum('nkj,nji->nki',Q,coefficients)
    error=float(np.max(np.abs(np.einsum('nki,nkj->nji',stencil,P)-target)))
    if error>5e-9:raise ValueError(f'Gradient polynomial reproduction failed:{error}')
    return indices.astype(np.int32),stencil,dict(polynomial_error=error,
        maximum_condition=float(np.max(condition)),degree=degree,neighbors=k,weight_exponent=c.nl.weight_exponent)


def gradient(values,indices,stencil):
    import numpy as np
    if (values.ndim!=3 or values.shape[1]!=len(indices) or indices.ndim!=2
            or stencil.shape!=(*indices.shape,2)):
        raise ValueError('Gradient needs(time,nodes,components), aligned neighbors and two derivative rows')
    return np.einsum('nkd,tnkc->tncd',stencil,values[:,indices]-values[:,:,None])


def viscous_divergence(tau,indices,stencil):
    import numpy as np
    if tau.shape[-1]!=3:raise ValueError('Symmetric2Dstress needs(xx,xy,yy)components')
    d=gradient(tau,indices,stencil)
    return np.stack((d[...,0,0]+d[...,1,1],d[...,1,0]+d[...,2,1]),axis=-1)


def probe(run,output):
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,fields,spatial_field_jets
    from examples.channel_nsch2d.problem import quadrature
    from docs.tasks.channel_trained_operator_probe import local_laplacian
    run=Path(run);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'probe_source.py')
    def record(stage):
        (output/'progress.json').write_text(json.dumps(dict(stage=stage))+'\n');print(stage,flush=True)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if jax.default_backend()!='cpu':raise RuntimeError('Frozen GMLS stress diagnostic must remain CPU-only')
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key));encoded=encode(model,data)
    times=jnp.asarray([.025,.5,1.,6.],dtype=c.train.compute_dtype)
    @eqx.filter_jit
    def jets(net,context,x):return spatial_field_jets(net,context,x,times,data['amplitudes'],c)
    def evaluate(points):
        rows=[];chunk=2048
        for start in range(0,len(points),chunk):
            q=jnp.asarray(points[start:start+chunk],dtype=c.train.compute_dtype);size=len(q)
            q=jnp.concatenate((q,jnp.repeat(q[-1:],chunk-size,axis=0)))
            rows.append([np.asarray(a,dtype=np.float64)[:,:size] for a in jets(model,encoded,q)])
        return [np.concatenate([r[i] for r in rows],axis=1) for i in range(3)]
    ix=np.unique(np.rint(np.linspace(1,c.sample.Nx-1,32)).astype(int))
    iy=np.unique(np.rint(np.linspace(1,c.sample.Ny-1,32)).astype(int))
    ii,jj=np.meshgrid(ix,iy,indexing='ij');ids=(ii*(c.sample.Ny+1)+jj).ravel()
    if len(ids)!=1024:raise ValueError('Expected the same1024interior velocity queries')
    exact=eqx.filter_jit(lambda net,context,x:local_laplacian(lambda q:fields(net,context,q,times,data['amplitudes'],c)[...,2:4],x))
    record('exact_AD_velocity_Laplacian');primal,lap=[np.asarray(a,dtype=np.float64) for a in exact(model,encoded,data['points'][ids])]
    record('original_grid_exact_viscous_stress');coarse=evaluate(raw['points']);replay=0.
    for n,time in enumerate(np.asarray(times)):
        match=np.flatnonzero(np.isclose(saved['time'],time,atol=2e-7,rtol=0))
        if len(match):replay=max(replay,float(np.max(np.abs(coarse[0][n]-saved['fields'][match[0]]))))
    def tau(values):
        _,dx,dy=values;nu=c.physical.viscosity
        return nu*np.stack((2*dx[...,2],dy[...,2]+dx[...,3],2*dy[...,3]),axis=-1)
    coarse_tau=tau(coarse);variants={};reports={}
    for label,degree,k,exponent in [('degree3',3,36,16.),('degree4',4,48,12.)]:
        selected=ConfigDict(c.to_dict());selected.nl.degree=degree;selected.nl.neighbors=k;selected.nl.weight_exponent=exponent
        indices,stencil,report=gradient_stencils(raw['points'],selected)
        variants[label]=viscous_divergence(coarse_tau,indices,stencil)[:,ids];reports[label]=report
    fine=ConfigDict(c.to_dict());fine.sample.Nx*=2;fine.sample.Ny*=2
    points,_=quadrature(raw['amplitudes'],fine,fine.sample.Nx,fine.sample.Ny)
    refined_ids=((2*ii)*(fine.sample.Ny+1)+2*jj).ravel()
    np.testing.assert_array_equal(points[refined_ids].astype(np.float32),raw['points'][ids].astype(np.float32))
    record('refined_grid_exact_viscous_stress');refined=evaluate(points);fine_tau=tau(refined)
    indices,stencil,report=gradient_stencils(points,fine)
    variants['refined_degree3']=viscous_divergence(fine_tau,indices,stencil)[:,refined_ids];reports['refined_degree3']=report
    primal_gap=max(float(np.max(np.abs(coarse[0][:,ids,2:4]-primal))),float(np.max(np.abs(refined[0][:,refined_ids,2:4]-primal))))
    if max(primal_gap,replay)>2e-5:raise ValueError(f'Frozen physical/replay gate failed:{primal_gap},{replay}')
    old=c.physical.viscosity*np.sum(raw['laplacian'][None,ids,:,None]*(coarse[0][:,raw['neighbors'][ids],2:4]-coarse[0][:,ids,None,2:4]),axis=2)
    w=(raw['transport_measure']*raw['interior'])[ids];w/=w.sum();rms=lambda a:float(np.sqrt(np.sum(a*a*w[:,None])))
    rows=[]
    for n,time in enumerate(np.asarray(times)):
        target=c.physical.viscosity*lap[n];norm=rms(target)
        if norm<=0:raise ValueError('Nonzero exact viscous norm required')
        rows.append(dict(time=float(time),viscous_AD_rms=norm,GMLS_Laplacian_relative=rms(old[n]-target)/norm,
            **{label:rms(value[n]-target)/norm for label,value in variants.items()}))
    np.savez_compressed(output/'frozen_fields.npz',time=np.asarray(times),points=raw['points'],
        fields=coarse[0],tau=coarse_tau,subset=ids,exact_velocity_laplacian=lap,
        refined_points=points,refined_fields=refined[0],refined_tau=fine_tau)
    result=dict(status='frozen_GMLS_exact_stress_consistency_completed',run=str(run),records=rows,
        saved_GPU_replay_gap=replay,primal_gap=primal_gap,queries=len(ids),operators=reports,
        scope='Frozen NN derivative consistency only; no learned stresses, new training/reference, BC change or L2 improvement claim',
        summation='Host FP64 GMLS weights and sums with CPU neural fields; not a GPU reduction emulation')
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');record('comparison_completed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.output)
