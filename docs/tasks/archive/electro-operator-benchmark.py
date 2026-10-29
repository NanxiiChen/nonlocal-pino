"""Actual trained-model forward+time-JVP timing: shared vs repeated space."""
import json
from pathlib import Path
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.electropolish2d.operator_training import _build_network

root=Path('/root/autodl-tmp/tf-logs/electropolish2d');records={}
with np.load('/root/autodl-tmp/data/electropolish2d/operator_random16_v1/conditions.npz') as f:
    images=f['images'];initial=f['initial'];points=f['points']
for kind in ('cross','film'):
    run=root/f'20261028-electro-operator-{kind}-01'
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c.operator,c.operator.image_shape,jax.random.key(0)))
    ids=np.random.default_rng(0).choice(len(points),1024,replace=False)
    image=jnp.asarray(images[16]);x=jnp.asarray(points[ids],dtype=jnp.float32)
    local=jnp.asarray(initial[16,ids]);ts=jnp.asarray([.1,.35,.6,1.],dtype=jnp.float32)
    xx=jnp.tile(x,(len(ts),1));tt=jnp.repeat(ts,len(x));ll=jnp.tile(local,(len(ts),1))
    split=eqx.filter_jit(lambda n,im,x,t,l:jax.jvp(lambda v:n(im,x,v,l),(t,),(jnp.ones_like(t),)))
    paired=eqx.filter_jit(lambda n,im,x,t,l:jax.jvp(lambda v:n.paired(im,x,v,l),(t,),(jnp.ones_like(t),)))
    outputs={};times={}
    for name,fn,args in [('shared',split,(net,image,x,ts,local)),('repeated',paired,(net,image,xx,tt,ll))]:
        compile_start=time.perf_counter();out=fn(*args);jax.block_until_ready(out)
        compile_seconds=time.perf_counter()-compile_start
        for _ in range(10):jax.block_until_ready(fn(*args))
        samples=[]
        for _ in range(50):
            start=time.perf_counter();out=fn(*args);jax.block_until_ready(out);samples.append(time.perf_counter()-start)
        outputs[name]=tuple(np.asarray(a).reshape(4,1024,2) for a in out)
        times[name]=dict(median_ms=1000*float(np.median(samples)),mean_ms=1000*float(np.mean(samples)),first_compile_and_call_seconds=compile_seconds)
    gaps=[float(np.max(abs(a-b))) for a,b in zip(outputs['shared'],outputs['repeated'])]
    # Record the failed strict FP32 probe separately; do not hide its discrepancies.
    relative_gaps=[float(np.linalg.norm(a-b)/np.linalg.norm(a))
                   for a,b in zip(outputs['shared'],outputs['repeated'])]
    with jax.enable_x64(True):
        promoted=jax.tree.map(lambda a:a.astype(jnp.float64) if eqx.is_inexact_array(a) else a,net)
        args64=(promoted,image.astype(jnp.float64),x.astype(jnp.float64),ts.astype(jnp.float64),local.astype(jnp.float64))
        flat64=(promoted,image.astype(jnp.float64),xx.astype(jnp.float64),tt.astype(jnp.float64),ll.astype(jnp.float64))
        out64=split(*args64);pair64=paired(*flat64);jax.block_until_ready(pair64)
        gaps64=[float(np.max(np.abs(np.asarray(a).reshape(4,1024,2)-np.asarray(b).reshape(4,1024,2)))) for a,b in zip(out64,pair64)]
        for a,b in zip(out64,pair64):
            np.testing.assert_allclose(np.asarray(a).reshape(4,1024,2),np.asarray(b).reshape(4,1024,2),rtol=1e-9,atol=1e-10)
    records[kind]=dict(times=times,raw_output_max_abs_difference=gaps[0],raw_time_jvp_max_abs_difference=gaps[1],fp32_relative_l2_differences=relative_gaps,fp64_max_abs_differences=gaps64,
        repeated_over_shared_speed_ratio=times['repeated']['median_ms']/times['shared']['median_ms'],
        points=1024,time_points=4,scope='one condition; forward and time JVP only; excludes optimizer and PDE',
        parameters=sum(a.size for a in jax.tree.leaves(net) if eqx.is_array(a)))
    print(kind,json.dumps(records[kind]),flush=True)
Path('docs/tasks/archive/electro-operator-benchmark.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
