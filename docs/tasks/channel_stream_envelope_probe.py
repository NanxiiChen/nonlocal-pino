"""Frozen cold-model inlet-envelope conditioning; no fitting or reference solve."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.network import build_network,encode,decode,lift,lifting_gates,time_jets
from examples.channel_nsch2d.problem import walls,flow_rate
from examples.channel_nsch2d.losses import gmls_first_jets,gmls_phase_advection,residual_components


def probe(run,output):
    run,output=Path(run),Path(output)
    if not run.is_absolute()or not output.is_absolute():raise ValueError('Absolute source/output paths required')
    if output.exists():raise FileExistsError(output)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if ('stream_lifting'in c or 'time_derivative'not in c or 'derivatives'not in c
            or c.lifting.kind!='normalized' or c.train.compute_dtype!='float32'):
        raise ValueError('Require the explicit original-envelope FP32 JVP baseline')
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_initial.eqx',build_network(c,key))
    model=jax.tree.map(lambda a:a.astype(jnp.float64)if eqx.is_inexact_array(a)else a,model)
    with np.load(run/'evaluation_inputs.npz',allow_pickle=False)as f:
        raw={k:f[k]for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer)else jnp.float64)for k,v in raw.items()}
    c.train.compute_dtype='float64';encoded=eqx.filter_jit(encode)(model,data);times=jnp.asarray([.1,1.,3.],dtype=jnp.float64)
    cases=[];parity=[]

    def gate(x,kind):
        if kind=='quadratic':return (x/c.geometry.length)**2
        if kind=='normalized_inlet':return (-jnp.expm1(-x/c.lifting.inlet_length_scale))**2
        if kind=='normalized_height':return (-jnp.expm1(-x/c.geometry.height))**2
        raise ValueError('Unknown frozen envelope')

    scales=jnp.asarray([c.loss.phase_scale,c.loss.chemical_scale,c.loss.momentum_scale,c.loss.momentum_scale])
    measure=data['transport_measure']*data['interior'];x=np.asarray(data['points'][:,0])
    masks={'all':np.ones(len(x),dtype=bool),'inlet_x_below_0p2':x<.2,'remaining':x>=.2}
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    for kind in ('quadratic','normalized_inlet','normalized_height'):
        def potential(net,e,d,points,ts):
            value=decode(net,net.prepare_encoded(e,points),ts,c)
            physical=lift(value,points,ts,d['amplitudes'],c)
            b,t=walls(points[:,0],d['amplitudes'],c,xp=jnp);eta=(points[:,1]-b)/(t-b)
            _,time=lifting_gates(points,ts,c)
            stream=flow_rate(ts,c,xp=jnp)[:,None]*(3*eta**2-2*eta**3)[None]
            stream+=c.loss.stream_scale*time[:,None]*(16*(eta*(1-eta))**2*gate(points[:,0],kind))[None]*value[...,2]
            return physical.at[...,2].set(stream)

        def physical(net,e,d,ts):
            points=d['points'];ex=jnp.broadcast_to(jnp.array([1.,0.]),points.shape);ey=ex[:,::-1]
            value,dx=jax.jvp(lambda p:potential(net,e,d,p,ts),(points,),(ex,))
            _,dy=jax.jvp(lambda p:potential(net,e,d,p,ts),(points,),(ey,))
            return jnp.stack((value[...,0],value[...,1],dy[...,2],-dx[...,2],value[...,3]),axis=-1)

        @eqx.filter_jit
        def evaluate(net,e,d,ts):
            value,rate=jax.jvp(lambda t:physical(net,e,d,t),(ts,),(jnp.ones_like(ts),))
            dx,dy=gmls_first_jets(value,value,d,boundary=False)
            lap=jnp.sum(d['laplacian'][None,:,:,None]*(value[:,d['neighbors'],:]-value[:,:,None,:]),axis=2)
            residual=residual_components(value,rate,dx,dy,lap,c,phase_advection=gmls_phase_advection(value,d))[...,:4]/scales
            return value,rate,residual,dx[...,2]+dy[...,3]

        value,rate,residual,div=[np.asarray(a)for a in evaluate(model,encoded,data,times)]
        if not all(np.isfinite(a).all()for a in (value,rate,residual,div)):raise ValueError('Nonfinite frozen conditioning probe')
        if kind!='normalized_height':
            canonical=ConfigDict(c.to_dict())
            if kind=='normalized_inlet':canonical.stream_lifting=dict(kind=kind)
            original,original_rate=eqx.filter_jit(lambda n,e,d,t:time_jets(n,e,d['points'],t,d['amplitudes'],canonical))(model,encoded,data,times)
            gap=max(float(np.max(np.abs(value-np.asarray(original)))),float(np.max(np.abs(rate-np.asarray(original_rate)))))
            if gap>2e-11:raise ValueError(f'Frozen mapping/canonical physicalfield/timeJVP gap: {gap}')
            parity.append(dict(kind=kind,maximum_primal_or_rate_gap=gap))
        rows=[];weight=np.asarray(measure)
        for label,mask in masks.items():
            w=weight*mask
            if w.sum()<=0:raise ValueError('Empty weighted diagnostic region')
            rows.append(dict(region=label,area_fraction=float(w.sum()/weight.sum()),
                mean_scaled_parts=np.mean(np.sum(residual**2*w[None,:,None],axis=1)/w.sum(),axis=0).tolist(),
                GMLS_divergence_RMS=float(np.sqrt(np.mean(np.sum(div**2*w[None],axis=1)/w.sum())))))
        momentum=np.sum(residual[...,2:4]**2,axis=-1).mean(0)*weight
        inlet=float(momentum[x<.2].sum()/momentum.sum())
        h3=float(jax.grad(jax.grad(jax.grad(lambda z:gate(z,kind))))(jnp.asarray(0.)))
        expected=0. if kind=='quadratic'else -6/(c.lifting.inlet_length_scale if kind=='normalized_inlet'else c.geometry.height)**3
        np.testing.assert_allclose(h3,expected,rtol=2e-14,atol=2e-14)
        np.savez_compressed(output/(kind+'.npz'),times=np.asarray(times),points=raw['points'],residual=residual,GMLS_divergence=div)
        case=dict(kind=kind,gate_third_derivative_at_inlet=h3,inlet_momentum_residual_fraction=inlet,regions=rows)
        cases.append(case);print(json.dumps(case),flush=True)
    report=dict(status='frozen_cold_envelope_conditioning_probe_completed',source_run=str(run),
        parameter_source='model_initial.eqx; exact saved FP32 values promoted to CPUFP64, no fitting',
        operator_source='evaluation_inputs.npz; exact stored coefficients promoted, no reconstruction',
        times=np.asarray(times).tolist(),domain_nodes=len(x),parts=['phase','chemical','momentum_x','momentum_y'],
        canonical_parity=parity,cases=cases,evolved_labels=0,FEM_solves=0,
        scope='Frozen initial-parameter conditioning and original trained GMLS-grid diagnostics; geometry-height gate is only a diagnostic, not a trained result, live-run change or continuum accuracy proof')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.output)
