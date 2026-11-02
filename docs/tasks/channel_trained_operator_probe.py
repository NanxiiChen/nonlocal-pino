"""Frozen trained-field GMLS versus exact AD; no trajectory/reference solve."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def local_laplacian(function,points):
    """Exact pointwise AD for query-independent fields (including psi-derived u)."""
    import jax
    import jax.numpy as jnp
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    value=function(points)
    _,xx=jax.jvp(lambda x:jax.jvp(function,(x,),(ex,))[1],(points,),(ex,))
    _,yy=jax.jvp(lambda x:jax.jvp(function,(x,),(ey,))[1],(points,),(ey,))
    return value,xx+yy


def probe(after_pid,progress_path,run,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'probe_source.py')
    def record(stage,**extra):
        (output/'progress.json').write_text(json.dumps(dict(stage=stage,**extra),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    record('waiting_for_finite_time_extension',pid=after_pid)
    while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    previous=json.loads(Path(progress_path).read_text())
    if previous['stage']=='online_training_and_input_replay_completed_unseen_validation_required':
        record('skipped_after_online_training',goal_complete=False);return
    if previous['stage']!='single_geometry_target_still_unmet':raise ValueError('Expected a complete failed accuracy assessment')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('A GPU job remains; no diagnostic overlap allowed')
    os.environ['JAX_PLATFORMS']='cuda'
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.network import build_network,encode,fields,potentials
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.problem import material
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    refined=ConfigDict(c.to_dict());refined.nl.degree=4;refined.nl.neighbors=48;refined.nl.weight_exponent=12.
    raw4,operator4=geometry_data(raw['amplitudes'],refined)
    for name in raw:
        if name not in ('neighbors','laplacian'):np.testing.assert_array_equal(raw[name],raw4[name])
    convert=lambda d:{k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in d.items()}
    data=convert(raw);data4=convert(raw4)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key));encoded=encode(model,data)
    ix=np.unique(np.rint(np.linspace(1,c.sample.Nx-1,32)).astype(int))
    iy=np.unique(np.rint(np.linspace(1,c.sample.Ny-1,32)).astype(int))
    ii,jj=np.meshgrid(ix,iy,indexing='ij');subset=(ii*(c.sample.Ny+1)+jj).ravel()
    if len(subset)!=1024:raise ValueError('Expected the declared32x32interior velocity queries')
    times=np.asarray([.025,.05,.1,.25,.5,1.,3.,6.],dtype=c.train.compute_dtype)
    @eqx.filter_jit
    def batch(net,e,d,d4,t):
        pm,lap_pm=local_laplacian(lambda x:potentials(net,e,x,t,d['amplitudes'],c)[...,:2],d['points'])
        full=fields(net,e,d['points'],t,d['amplitudes'],c)
        velocity,lap_v=local_laplacian(lambda x:fields(net,e,x,t,d['amplitudes'],c)[...,2:4],d['points'][subset])
        L=lambda value,ids,weights:jnp.sum(weights[None,:,:,None]*(value[:,ids,:]-value[:,:,None,:]),axis=2)
        g3=L(jnp.concatenate((pm,full[...,2:4]),axis=-1),d['neighbors'],d['laplacian'])
        g4=L(jnp.concatenate((pm,full[...,2:4]),axis=-1),d4['neighbors'],d4['laplacian'])
        gap=jnp.maximum(jnp.max(jnp.abs(full[...,:2]-pm)),jnp.max(jnp.abs(full[:,subset,2:4]-velocity)))
        return pm,lap_pm,lap_v,g3,g4,full,gap
    measure=raw['transport_measure']*raw['interior'];measure=measure/measure.sum()
    selected=measure[subset];selected=selected/selected.sum();A,kappa=material(c)
    rms=lambda a,w:float(np.sqrt(np.sum(a*a*w.reshape((-1,)+(1,)*(a.ndim-1)))))
    relative=lambda a,b,w:rms(a-b,w)/rms(b,w)
    records=[];primal=0.;replay=0.
    record('evaluating_frozen_AD_and_both_operators',run=str(run))
    for offset in range(0,len(times),4):
        values=[np.asarray(a,dtype=np.float64) for a in batch(model,encoded,data,data4,jnp.asarray(times[offset:offset+4]))]
        pm,exact,exact_v,g3,g4,full,gap=values;primal=max(primal,float(gap))
        for n,t in enumerate(times[offset:offset+4]):
            match=np.flatnonzero(np.isclose(saved['time'],t,atol=2e-7,rtol=0))
            if len(match):replay=max(replay,float(np.max(np.abs(full[n]-saved['fields'][match[0]]))))
            base=pm[n,:,1]-A*(pm[n,:,0]**3-pm[n,:,0]);row=dict(time=float(t))
            row['chemical_rms_AD']=rms(base+kappa*exact[n,:,0],measure)
            row['AD_laplacian_rms_phi']=rms(exact[n,:,0],measure)
            row['AD_laplacian_rms_mu']=rms(exact[n,:,1],measure)
            for label,discrete in (('degree3',g3[n]),('degree4',g4[n])):
                row[label]=dict(laplacian_phi_relative=relative(discrete[:,0],exact[n,:,0],measure),
                    laplacian_mu_relative=relative(discrete[:,1],exact[n,:,1],measure),
                    chemical_discretization_shift_rms=rms(kappa*(discrete[:,0]-exact[n,:,0]),measure),
                    chemical_rms=rms(base+kappa*discrete[:,0],measure),
                    phase_diffusion_shift_rms=rms(c.physical.mobility*(discrete[:,1]-exact[n,:,1]),measure),
                    velocity_laplacian_relative=relative(discrete[subset,2:4],exact_v[n],selected),
                    viscous_discretization_shift_rms=rms(c.physical.viscosity*(discrete[subset,2:4]-exact_v[n]),selected))
            records.append(row)
        record('evaluating_frozen_AD_and_both_operators',completed_times=offset+4,total_times=len(times))
    if not np.isfinite(np.asarray([r['chemical_rms_AD'] for r in records])).all() or max(primal,replay)>2e-5:
        raise ValueError(f'Frozen trained-operator primal replay failed: {primal},{replay}')
    report=dict(status='trained_field_operator_comparison_completed',run=str(run),records=records,
        primal_gap=primal,saved_GPU_replay_gap=replay,grid=[c.sample.Nx,c.sample.Ny],velocity_queries=len(subset),
        degree3=c.nl.to_dict(),degree4=refined.nl.to_dict(),degree4_construction=operator4,
        scope='Operator consistency of this fixed NN, not a native coupled trajectory, continuum error bound or trained improvement')
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    record('comparison_completed',comparison=str(output/'comparison.json'),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--progress',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.after_pid,a.progress,a.run,a.output)
