"""Unweighted canonical timeJVP/GMLS residuals at fresh continuous probe times."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.losses import make_time_parts_fn,part_names
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.problem import validate_config
from nonlocal_pinn.training.runtime import configure_gpu_policy


def diagnose(run,output,time_samples,probe_seed):
    run,output=Path(run),Path(output)
    c=ConfigDict(json.loads((run/'config.json').read_text()));validate_config(c)
    if 'time_derivative' not in c or c.time_derivative.kind!='jvp':
        raise ValueError('Continuous residual audit requires explicit timeJVP')
    size=c.train.microbatch.size
    if (isinstance(time_samples,bool) or not isinstance(time_samples,int)
            or time_samples<1 or time_samples%size or isinstance(probe_seed,bool)
            or not isinstance(probe_seed,int) or probe_seed<0):
        raise ValueError('Positive probe count divisible by microbatch and nonnegative integer seed required')
    if output.exists():raise FileExistsError(output)
    summary=json.loads((run/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('Completed saved model required')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    rng=np.random.default_rng(probe_seed)
    times=((np.arange(time_samples)+rng.random(time_samples))*c.physical.terminal_time/time_samples).astype(c.train.compute_dtype)
    if np.any(np.diff(times)<=0) or times[0]<=0 or times[-1]>=c.physical.terminal_time:
        raise ValueError('Probe times must be strictly ordered and inside the physical horizon')
    per_time=make_time_parts_fn(c)
    evaluate=eqx.filter_jit(lambda n,d,t:per_time(n,d,t))
    inputs=[]
    with np.load(run/'evaluation_inputs.npz',allow_pickle=False)as f:
        inputs.append(('nominal',{k:f[k]for k in f.files}))
    with np.load(run/'known_inputs_final.npz',allow_pickle=False)as f:
        inputs.append(('final_pool_slot0',{k:f[k][0]for k in f.files}))
    output.mkdir(parents=True);cases=[]
    for label,raw in inputs:
        data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer)
            else c.train.compute_dtype)[None]for k,v in raw.items()}
        parts=[]
        for start in range(0,time_samples,size):
            value=np.asarray(evaluate(model,data,jnp.asarray(times[start:start+size])))
            if value.shape!=(1,size,len(part_names(c))) or not np.isfinite(value).all():
                raise ValueError(f'Nonfinite or misaligned canonical JVP diagnostics: {label}, block{start}')
            parts.append(value[0])
            if start==0 or start+size==time_samples:
                print(json.dumps(dict(case=label,completed_times=start+size,total_times=time_samples)),flush=True)
        parts=np.concatenate(parts)
        np.savez_compressed(output/(label+'-losses.npz'),times=times,parts=parts,amplitudes=raw['amplitudes'])
        case=dict(label=label,amplitudes=raw['amplitudes'].tolist(),domain_nodes=len(raw['points']),
            time_samples=time_samples,mean_parts=parts.mean(0).tolist(),maximum_parts=parts.max(0).tolist(),
            mean_unweighted_total=float(parts.sum(-1).mean()),source=str(output/(label+'-losses.npz')))
        cases.append(case)
    report=dict(status='canonical_unweighted_continuous_time_probes_completed',run=str(run),
        time_derivative=c.time_derivative.to_dict(),time_samples=time_samples,probe_seed=probe_seed,
        part_names=list(part_names(c)),cases=cases,reference_labels_used=0,
        spatial_operator=c.nl.to_dict(),
        scope='Canonical unweighted timeJVP PDE+BC losses with the saved spatial operator on the full trained grid, at independent continuous times; no causal discount or continuum convergence proof')
    if 'chemical_residual_scale'in c.loss:
        report['chemical_scaling']=dict(output_amplitude=c.loss.chemical_scale,
            residual_normalizer=c.loss.chemical_residual_scale,
            scope='Reported chemical MSE uses this explicit row normalizer; unweighted means no causal discount')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--time-samples',required=True,type=int);p.add_argument('--probe-seed',required=True,type=int)
    a=p.parse_args();diagnose(a.run,a.output,a.time_samples,a.probe_seed)
