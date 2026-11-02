"""CPU-only frozen constitutive residual allocation, with GPU agreement gates."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def probe(run,objective_record,output):
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    # Keep this independent read-only probe from taking the compiler's CPU set.
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[-4:])
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,decode,lift
    from examples.channel_nsch2d.problem import material
    run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError(output)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    np.testing.assert_array_equal(saved['points'],np.asarray(data['points']))
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    encoded=encode(model,data);h=model.prepare_encoded(encoded,data['points'])
    evaluate=eqx.filter_jit(lambda net,t:lift(decode(net,h,t,c),data['points'],t,data['amplitudes'],c)[...,:2])
    count=int(round(c.physical.terminal_time/c.time_discretization.dt))
    if count%8:raise ValueError('Frozen diagnostic requires complete eight-time batches')
    nodes=np.arange(1,count+1,dtype=c.train.compute_dtype)*np.asarray(c.time_discretization.dt,dtype=c.train.compute_dtype)
    measure=np.asarray(data['transport_measure'] if 'phase_transport' in c else data['volumes'],dtype=np.float64)*raw['interior']
    measure=measure/measure.sum()
    lap=np.asarray(data['laplacian'],dtype=np.float64);neighbors=raw['neighbors'];A,kappa=material(c)
    initial_support=np.abs(np.asarray(data['domain'])[:,2])<.999  # Initial transition band, not the whole droplet.
    rows=[];gap=0.
    for offset in range(0,count,8):
        values=np.asarray(evaluate(model,jnp.asarray(nodes[offset:offset+8])),dtype=np.float64)
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite frozen chemical probe')
        phi,mu=values[...,0],values[...,1]
        Lphi=np.sum(lap[None]*(phi[:,neighbors]-phi[:,:,None]),axis=-1)
        target=A*(phi**3-phi)-kappa*Lphi;residual=mu-target
        square=(residual/c.loss.chemical_scale)**2
        for i,t in enumerate(nodes[offset:offset+8]):
            matches=np.flatnonzero(np.isclose(saved['time'],t,atol=2e-7,rtol=0))
            if len(matches):gap=max(gap,float(np.max(np.abs(values[i]-saved['fields'][matches[0],:,:2]))))
            rows.append(dict(time=float(t),chemical_loss=float(np.sum(square[i]*measure)),
                interface_loss=float(np.sum(square[i]*measure*(np.abs(phi[i])<.9))),
                initial_support_loss=float(np.sum(square[i]*measure*initial_support)),
                chemical_rms=float(np.sqrt(np.sum(mu[i]**2*measure))),
                constitutive_rhs_rms=float(np.sqrt(np.sum(target[i]**2*measure))),
                residual_max=float(np.max(np.abs(residual[i])))))
    baseline=json.loads(Path(objective_record).read_text().splitlines()[0])
    if baseline['evaluation']!=1 or baseline['time_nodes']!=count:raise ValueError('First complete objective support mismatch')
    actual=float(np.mean([r['chemical_loss'] for r in rows]));reference=baseline['parts']['chemical']
    relative=abs(actual-reference)/reference
    total=sum(r['chemical_loss'] for r in rows)
    report=dict(status='passed' if gap<2e-5 and relative<.05 else 'CPU_agreement_failed',run=str(run),
        cpu_gpu_primal_max_gap=gap,chemical_loss_CPU=actual,chemical_loss_actual_GPU=reference,
        chemical_loss_relative_gap=relative,time_nodes=count,
        cumulative_chemical_loss_fractions={str(end):sum(r['chemical_loss'] for r in rows if r['time']<=end+1e-6)/total
            for end in (.1,.25,.5,1.,2.)},
        full_initial_support_fraction=sum(r['initial_support_loss'] for r in rows)/total,
        initial_support_definition='abs(phi0)<0.999: initial transition band only',
        full_interface_fraction=sum(r['interface_loss'] for r in rows)/total,
        records=rows,scope='Frozen original model, actual operators; reporting allocation only, not a loss change or unique-cause proof')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
    if report['status']!='passed':raise ValueError('Frozen CPU constitutive diagnostic did not match actual GPU objective')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--objective-record',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.objective_record,a.output)
