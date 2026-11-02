"""Fixed-network precision and physical-grid Laplacian consistency, no training."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def probe(run,output):
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Finish existing GPU work before the frozen-grid diagnostic')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'probe_source.py')
    def record(stage,**extra):
        (output/'progress.json').write_text(json.dumps(dict(stage=stage,**extra),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    os.environ['JAX_PLATFORMS']='cuda'
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.network import build_network,encode,fields
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.problem import material
    from docs.tasks.channel_trained_operator_probe import local_laplacian
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    if c.train.compute_dtype!='float32':raise ValueError('This paired precision diagnostic requires the FP32 parent')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    high=ConfigDict(c.to_dict());high.nl.degree=4;high.nl.neighbors=48;high.nl.weight_exponent=12.
    record('constructing_known_input_stencils')
    raw4,report4=geometry_data(raw['amplitudes'],high)
    fine=ConfigDict(high.to_dict());fine.sample.Nx*=2;fine.sample.Ny*=2
    raw_fine,report_fine=geometry_data(raw['amplitudes'],fine)
    for name in raw:
        if name not in ('neighbors','laplacian'):np.testing.assert_array_equal(raw[name],raw4[name])
    ix=np.unique(np.rint(np.linspace(1,c.sample.Nx-1,32)).astype(int))
    iy=np.unique(np.rint(np.linspace(1,c.sample.Ny-1,32)).astype(int))
    ii,jj=np.meshgrid(ix,iy,indexing='ij');subset=(ii*(c.sample.Ny+1)+jj).ravel()
    refined_subset=((2*ii)*(fine.sample.Ny+1)+2*jj).ravel()
    if len(subset)!=1024:raise ValueError('Expected the declared32x32interior query set')
    np.testing.assert_array_equal(raw['points'][subset].astype(np.float32),raw_fine['points'][refined_subset].astype(np.float32))
    times=np.asarray([.025,.5,1.,6.],dtype=np.float32)
    data32={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else 'float32') for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model32=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key));encoded32=encode(model32,data32)

    def functions(config):
        @eqx.filter_jit
        def jets(net,encoded,points,t,amplitudes):
            return local_laplacian(lambda x:fields(net,encoded,x,t,amplitudes,config)[...,:4],points)
        @eqx.filter_jit
        def values(net,encoded,points,t,amplitudes):return fields(net,encoded,points,t,amplitudes,config)
        return jets,values

    def full_values(function,net,encoded,points,t,amplitudes):
        rows=[];chunk=8192
        for start in range(0,len(points),chunk):
            part=points[start:start+chunk];count=len(part)
            padded=jnp.concatenate((part,jnp.repeat(part[-1:],chunk-count,axis=0)),axis=0)
            rows.append(np.asarray(function(net,encoded,padded,t,amplitudes))[:,:count])
        return np.concatenate(rows,axis=1)

    def laplacian(value,payload,indices):
        adjacent=value[:,payload['neighbors'][indices],:]-value[:,indices,None,:]
        return np.sum(payload['laplacian'][None,indices,:,None]*adjacent,axis=2)

    record('evaluating_original_FP32')
    jets32,values32=functions(c)
    primal32,exact32=[np.asarray(a,dtype=np.float64) for a in jets32(model32,encoded32,data32['points'][subset],jnp.asarray(times),data32['amplitudes'])]
    full32=full_values(values32,model32,encoded32,data32['points'],jnp.asarray(times),data32['amplitudes']).astype(np.float64)
    rounded={k:np.asarray(v) for k,v in data32.items()}
    discrete32=laplacian(full32[...,:4],rounded,subset)
    replay=0.
    for n,t in enumerate(times):
        match=np.flatnonzero(np.isclose(saved['time'],t,atol=2e-7,rtol=0))
        if len(match):replay=max(replay,float(np.max(np.abs(full32[n]-saved['fields'][match[0]]))))

    record('evaluating_promoted_FP64_with_frozen_original_context')
    jax.config.update('jax_enable_x64',True)
    c64=ConfigDict(c.to_dict());c64.train.compute_dtype='float64'
    model64=jax.tree.map(lambda a:a.astype(jnp.float64) if eqx.is_inexact_array(a) else a,model32)
    data64={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else 'float64') for k,v in rounded.items()}
    encoded64=encode(model64,data64);jets64,values64=functions(c64)
    t64=jnp.asarray(times,dtype=jnp.float64)
    primal64,exact64=[np.asarray(a) for a in jets64(model64,encoded64,data64['points'][subset],t64,data64['amplitudes'])]
    full64=full_values(values64,model64,encoded64,data64['points'],t64,data64['amplitudes'])
    fine_points=jnp.asarray(raw_fine['points'].astype(np.float32),dtype=jnp.float64)
    fine_values=full_values(values64,model64,encoded64,fine_points,t64,data64['amplitudes'])
    primal_gap=max(float(np.max(np.abs(full32[...,:4]-full64[...,:4]))),
        float(np.max(np.abs(primal32-primal64))),
        float(np.max(np.abs(full64[:,subset,:4]-primal64))),
        float(np.max(np.abs(fine_values[:,refined_subset,:4]-primal64))))
    if not np.isfinite(exact64).all() or max(primal_gap,replay)>2e-5:
        raise ValueError(f'Fixed physical field/precision/query replay failed: {primal_gap},{replay}')
    discrete=dict(original_FP32=discrete32,
        original_degree3_FP64=laplacian(full64[...,:4],raw,subset),
        original_degree4_FP64=laplacian(full64[...,:4],raw4,subset),
        refined_degree4_FP64=laplacian(fine_values[...,:4],raw_fine,refined_subset))
    measure=(raw['transport_measure']*raw['interior'])[subset];measure/=measure.sum()
    rms=lambda x:float(np.sqrt(np.sum(measure.reshape((-1,)+(1,)*(x.ndim-1))*x*x)))
    def relative(a,b):
        scale=rms(b)
        if not np.isfinite(scale) or scale<=0:raise ValueError('A finite nonzero exact derivative norm is required')
        return rms(a-b)/scale
    A,kappa=material(c);records=[]
    for n,t in enumerate(times):
        row=dict(time=float(t),AD_FP32_vs_FP64={})
        for name,index in (('phi',0),('mu',1),('velocity',slice(2,4))):
            row['AD_FP32_vs_FP64'][name]=relative(exact32[n,:,index],exact64[n,:,index])
        for name,value in discrete.items():
            exact=exact32[n] if name=='original_FP32' else exact64[n]
            primal=primal32[n] if name=='original_FP32' else primal64[n]
            base=primal[:,1]-A*(primal[:,0]**3-primal[:,0])
            row[name]=dict(phi_laplacian_relative=relative(value[n,:,0],exact[:,0]),
                mu_laplacian_relative=relative(value[n,:,1],exact[:,1]),
                velocity_laplacian_relative=relative(value[n,:,2:4],exact[:,2:4]),
                chemical_rms=rms(base+kappa*value[n,:,0]),chemical_rms_AD=rms(base+kappa*exact[:,0]))
        records.append(row)
    report=dict(status='fixed_network_precision_and_grid_comparison_completed',run=str(run),records=records,
        queries=len(subset),original_grid=[c.sample.Nx,c.sample.Ny],refined_grid=[fine.sample.Nx,fine.sample.Ny],
        primal_gap=primal_gap,saved_GPU_replay_gap=replay,degree4_construction=report4,refined_construction=report_fine,
        context='Original training nodes, encoder inputs and quadrature held fixed in both grids; FP64 promotes the original FP32 parameter/input values',
        scope='Consistency on a frozen learned field; no native trajectory, reference solution, training improvement or continuum-error claim')
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    record('comparison_completed',comparison=str(output/'comparison.json'),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.output)
