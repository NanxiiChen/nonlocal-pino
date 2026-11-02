"""Frozen model versus an explicitly refinement-validated dense local FEM set."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(run,validation):
    import os
    # The waiting/FEM orchestration parent is CPU-only. This separate frozen
    # evaluation starts only after FEM exits, and explicitly uses the GPU.
    os.environ['JAX_PLATFORMS']='cuda'
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.problem import identity
    from examples.channel_nsch2d.network import build_network,encode,fields
    from examples.channel_nsch2d.data import geometry_data
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    gate=json.loads(Path(validation).read_text())
    if gate['status']!='passed' or gate['identity']!=json.loads(json.dumps(identity(c))):
        raise ValueError('Dense local FEM must pass matching physical refinement gates')
    if json.loads((run/'summary.json').read_text())['status']!='completed':raise ValueError('A complete frozen checkpoint is required')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    with np.load(gate['reference'],allow_pickle=False) as f:ref={k:f[k] for k in f.files}
    raw,_=geometry_data(ref['amplitudes'],c)
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    predict=eqx.filter_jit(lambda model:fields(model,encode(model,data),
        jnp.asarray(ref['points'],dtype=c.train.compute_dtype),jnp.asarray(ref['time'],dtype=c.train.compute_dtype),data['amplitudes'],c))
    prediction=np.asarray(predict(net),dtype=np.float64)
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite dense prediction')
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    target=ref['fields'];metrics={}
    for name,sl,begin in (('phi',slice(0,1),0),('velocity',slice(2,4),1),('pressure',slice(4,5),1),('chemical',slice(1,2),1)):
        metrics[name]=relative(prediction[begin:,...,sl],target[begin:,...,sl])
    metrics['terminal_phi']=relative(prediction[-1,:,0],target[-1,:,0])
    metrics['dispersed_phase']=relative((1-prediction[...,0])/2,(1-target[...,0])/2)
    metrics['joint_physical_fields']=relative(prediction[..., [0,2,3,4]],target[..., [0,2,3,4]])
    metrics['positive_time_joint_physical_fields']=relative(prediction[1:,..., [0,2,3,4]],target[1:,..., [0,2,3,4]])
    per_time={name:[relative(a,b) for a,b in zip(prediction[begin:,...,sl],target[begin:,...,sl],strict=True)]
        for name,sl,begin in (('phi',slice(0,1),0),('velocity',slice(2,4),1),('pressure',slice(4,5),1))}
    output=run/'dense-fem-prediction.npz'
    np.savez_compressed(output,fields=prediction,points=ref['points'],time=ref['time'],amplitudes=ref['amplitudes'],reference=target)
    report=dict(status='dense_FEM_evaluation_completed',run=str(run),reference=gate['reference'],
        reference_validation=str(validation),query_points=len(ref['points']),time=ref['time'].tolist(),
        metrics=metrics,per_time_relative_L2=per_time,
        joint_metric_scope='Raw all-time joint includes unconstrained initial pressure versus the FEM zero initialization; positive-time joint excludes it. Neither joint determines acceptance.',
        single_geometry_target_met=all(metrics[k]<.1 for k in ('phi','terminal_phi','velocity','pressure')),
        scope='Frozen checkpoint, one development geometry; independently refined local FEM, no training labels or unseen-geometry claim')
    (run/'dense-fem-evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--validation',required=True)
    a=p.parse_args();main(a.run,a.validation)
