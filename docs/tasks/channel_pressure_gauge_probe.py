"""Frozen physical outlet-traction gauge probe; no optimizer or reference-derived shift."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def probe(run,prediction_path,output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,spatial_field_jets
    from examples.channel_nsch2d.data import geometry_data
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    with np.load(prediction_path,allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    raw,_=geometry_data(saved['amplitudes'],c)
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    roots,w=np.polynomial.legendre.leggauss(64);weight=w/2
    outlet=np.column_stack((np.full(64,c.geometry.length),(roots+1)*c.geometry.height/2))
    indices=np.linspace(0,len(saved['points'])-1,24,dtype=int)
    query=jnp.asarray(np.concatenate((saved['points'][indices],outlet)),dtype=c.train.compute_dtype)
    times=jnp.asarray(saved['time'],dtype=c.train.compute_dtype)
    jets=eqx.filter_jit(lambda model:spatial_field_jets(model,encode(model,data),query,times,data['amplitudes'],c))
    value,dx,dy=[np.asarray(v,dtype=np.float64) for v in jets(net)]
    if any(not np.isfinite(v).all() for v in (value,dx,dy)):raise FloatingPointError('Nonfinite gauge probe fields')
    gap=float(np.max(np.abs(value[:,:24]-saved['fields'][:,indices])))
    if gap>2e-5:raise ValueError(f'CPU/GPU physical replay failed: {gap}')
    defect=2*c.physical.viscosity*dx[:,24:,2]-value[:,24:,4]
    correction=np.sum(defect*weight[None,:],axis=1)
    before=saved['fields'][...,4];target=saved['reference'][...,4]
    after=before+correction[:,None]
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    report=dict(status='frozen_physical_gauge_diagnostic_completed',run=str(run),
        primal_CPU_GPU_max_gap=gap,quadrature_points=64,pressure_shift=correction.tolist(),
        pressure_before=relative(before[1:],target[1:]),pressure_after=relative(after[1:],target[1:]),
        positive_time_before=[relative(a,b) for a,b in zip(before[1:],target[1:],strict=True)],
        positive_time_after=[relative(a,b) for a,b in zip(after[1:],target[1:],strict=True)],
        maximum_projected_mean_traction=float(np.max(np.abs(np.sum((defect-correction[:,None])*weight[None,:],axis=1)))),
        scope='Physical traction-derived constant shift only; no model/optimizer change or reference-derived gauge; not adopted')
    with Path(output).open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--prediction',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.prediction,a.output)
