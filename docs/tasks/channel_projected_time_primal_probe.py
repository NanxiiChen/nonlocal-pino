"""Frozen physical-mu FP32 diagnostic: time-batch shape versus JVP arithmetic."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.network import build_network,encode
from examples.channel_nsch2d.projected_fields import make_domain_fn
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main(control,output):
    control,output=Path(control),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    c=ConfigDict(json.loads((control/'config.json').read_text()))
    c.chemical_constraint=dict(kind='interior_gmls_projection',boundary='decoder_trace',query='mapped_bilinear')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(control/'model_final.eqx',build_network(c,key))
    with np.load(control/'evaluation_inputs.npz',allow_pickle=False)as f:
        data={k:jnp.asarray(f[k])for k in f.files}
    full=jnp.linspace(0,6,13,dtype=jnp.float32);ids=np.array([1,6,12]);subset=full[ids]
    domain=make_domain_fn(c)
    def value(net,d,ts):return domain(net,encode(net,d),d,ts)
    def mapped(net,d,ts):
        e=encode(net,d)
        return jax.lax.map(lambda t:domain(net,e,d,t[None])[0],ts)
    reports=[]
    for name,fn in (('batched',value),('single_time_map',mapped)):
        primal=eqx.filter_jit(fn)
        jets=eqx.filter_jit(lambda net,d,ts:jax.jvp(lambda tt:fn(net,d,tt),(ts,),(jnp.ones_like(ts),)))
        a=np.asarray(primal(model,data,full));b=np.asarray(primal(model,data,subset))
        v,rate=[np.asarray(q)for q in jets(model,data,subset)]
        report=dict(kind=name,time_batch_gap=np.max(np.abs(a[ids]-b),axis=(0,1)).tolist(),
            JVP_same_time_primal_gap=np.max(np.abs(v-b),axis=(0,1)).tolist(),
            combined_batch_JVP_gap=np.max(np.abs(v-a[ids]),axis=(0,1)).tolist(),
            all_finite=bool(all(np.isfinite(q).all()for q in (a,b,v,rate))))
        reports.append(report);print(json.dumps(report),flush=True)
    result=dict(status='frozen_time_primal_diagnostic_completed',source_run=str(control),
        optimizer_updates=0,reference_labels_used=0,fitted_weights_exported=False,
        channels=['phi','mu','u','v','p'],nodes=len(data['points']),reports=reports)
    (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.control,a.output)
