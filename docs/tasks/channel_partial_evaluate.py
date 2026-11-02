"""Evaluate an explicitly named partial model without inventing optimizer recovery."""
from pathlib import Path
import argparse
import json
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network,encode,fields
from examples.channel_nsch2d.problem import identity,validate_config
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main(run,step):
    run=Path(run);checkpoint=run/f'ckpt_{step:06d}.eqx'
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    c=ConfigDict(json.loads((run/'config.json').read_text()));validate_config(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    gate=json.loads((Path(c.reference.root)/'reference-validation.json').read_text())
    if gate['status']!='passed' or gate['identity']!=json.loads(json.dumps(identity(c))):
        raise ValueError('Matching independent reference gate required')
    with np.load(gate['reference'],allow_pickle=False) as file:ref={k:file[k] for k in file.files}
    data,_=geometry_data(ref['amplitudes'],c)
    dtype=np.dtype(c.train.compute_dtype)
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else dtype) for k,v in data.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(checkpoint,build_network(c,key))
    evaluate=eqx.filter_jit(lambda n:fields(n,encode(n,data),jnp.asarray(ref['points'],dtype=dtype),
        jnp.asarray(ref['time'],dtype=dtype),data['amplitudes'],c))
    prediction=np.asarray(evaluate(net),dtype=np.float64)
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite partial physical prediction')
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics=dict(phi=relative(prediction[...,0],ref['fields'][...,0]),
        terminal_phi=relative(prediction[-1,:,0],ref['fields'][-1,:,0]),
        velocity=relative(prediction[1:,:,2:4],ref['fields'][1:,:,2:4]),
        pressure=relative(prediction[1:,:,4],ref['fields'][1:,:,4]))
    logged=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    expected=[r for r in logged if r['step']==step]
    if len(expected)!=1:raise ValueError('Partial checkpoint must have exactly one corresponding evaluation')
    metric_gap={k:abs(v-expected[0][k]) for k,v in metrics.items()}
    if max(metric_gap.values())>2e-6:raise ValueError(f'Partial logged metric replay failed: {metric_gap}')
    target=run/f'partial_{step:06d}.npz'
    np.savez_compressed(target,fields=prediction,points=ref['points'],time=ref['time'],
        amplitudes=ref['amplitudes'],reference=ref['fields'])
    report=dict(status='partial_checkpoint_metric_replay_passed',checkpoint=str(checkpoint),
        step=step,metrics=metrics,logged_metric_absolute_gap=metric_gap,
        phase_min=prediction[...,0].min(axis=1).tolist(),phase_max=prediction[...,0].max(axis=1).tolist(),
        terminal_negative_queries=int(np.sum(prediction[-1,:,0]<0)),prediction=str(target),
        scope='One development FEM geometry, partial model only; no optimizer recovery or complete-run/physical-balance claim')
    (run/f'partial_{step:06d}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--step',required=True,type=int);a=p.parse_args();main(a.run,a.step)
