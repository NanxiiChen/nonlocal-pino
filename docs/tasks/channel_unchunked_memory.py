"""Compile-only unchunked channel update memory; never call the executable."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.balance import group_names
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.problem import validate_config
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main(a):
    if (os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']!='false'
            or '--xla_gpu_autotune_level=0' not in os.environ['XLA_FLAGS'].split()):
        raise ValueError('Compile-only live-card diagnostic requires no preallocation and no GPU autotuning')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    run=Path(a.run);saved=json.loads((run/'config.json').read_text())
    c=ConfigDict(saved)
    if c.train.microbatch.to_dict()!=dict(kind='time',size=8) or c.sample.Nt!=40:
        raise ValueError('Expected actual40-time/8-microbatch control config')
    del c.train.microbatch
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    report=dict(status='compiling_only',run=str(run),jax_version=jax.__version__,
        execution_calls=0,optimizer_updates=0,scope='Compiler estimates only; GPU autotuning disabled; not a runtime peak/speed measurement',
        preallocate=os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'],XLA_FLAGS=os.environ['XLA_FLAGS'],plans=[])
    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    # Templates and numerical initial arrays are created on CPU. Only lowering
    # and compilation target CUDA; no executable is invoked on either device.
    with jax.default_device(jax.devices('cpu')[0]):
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
        model=eqx.tree_deserialise_leaves(run/'model_initial.eqx',build_network(c,key))
        optimizer=make_optimizer(c)
        state=eqx.tree_deserialise_leaves(run/'optimizer_initial.eqx',optimizer.init(eqx.filter(model,eqx.is_array)))
    host=lambda tree:jax.tree.map(lambda x:np.asarray(x) if eqx.is_array(x) else x,tree)
    model,state=host(model),host(state)
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:
        data={name:f[name][:c.operator.condition_batch].astype(f[name].dtype
            if np.issubdtype(f[name].dtype,np.integer) else c.train.compute_dtype) for name in f.files}
    gpu=jax.devices('gpu')[0]
    report['device']=gpu.device_kind
    report['card_total_MiB']=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total',
        '--format=csv,noheader,nounits'],text=True).strip())
    for nt in a.times:
        c.sample.Nt=nt;validate_config(c)
        loss=make_loss_fn(c)

        @eqx.filter_jit(backend='gpu')
        def step(net,opt,batch,times,weights,refresh,active):
            (value,parts),gradient=eqx.filter_value_and_grad(loss,has_aux=True)(net,batch,times)
            updates,opt=optimizer.update(gradient,opt,net)
            return eqx.apply_updates(net,updates),opt,value,parts,weights,jnp.zeros_like(weights)

        dtype=np.dtype(c.train.compute_dtype)
        times=(np.arange(nt,dtype=dtype)*int(round(c.physical.terminal_time/c.time_discretization.dt))//nt+1)*c.time_discretization.dt
        times=times.astype(dtype)
        start=time.perf_counter()
        executable=step.lower(model,state,data,times,np.ones(len(group_names(c)),dtype=dtype),
            np.array(False),np.ones(2,dtype=dtype)).compile()
        stats=executable.compiled.memory_analysis()
        if stats is None:raise RuntimeError('GPU compiler did not provide memory analysis')
        keys=('temp_size_in_bytes','argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes')
        sizes={name:int(getattr(stats,name)) for name in keys}
        if any(v<0 for v in sizes.values()) or sizes['temp_size_in_bytes']==0:
            raise ValueError('Incomplete compiler memory analysis for this full update')
        total=sizes['temp_size_in_bytes']+sizes['argument_size_in_bytes']+sizes['output_size_in_bytes']-sizes['alias_size_in_bytes']
        plan=dict(time_samples=nt,condition_batch=c.operator.condition_batch,
            nodes=c.sample.Nx*c.sample.Ny+c.sample.Nx+c.sample.Ny+1,
            compile_seconds=time.perf_counter()-start,**sizes,estimated_total_bytes=total,
            estimated_total_GiB=total/2**30,exceeds_physical_card=total>report['card_total_MiB']*2**20)
        report['plans'].append(plan);save();print(json.dumps(plan),flush=True)
    report['status']='compile_only_memory_plans_complete';save()
    root=Path(__file__).resolve().parents[2]
    for name in ('docs/tasks/channel_unchunked_memory.py','examples/channel_nsch2d/losses.py',
                 'examples/channel_nsch2d/network.py','examples/channel_nsch2d/temporal.py'):
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--times',required=True,type=int,nargs='+')
    a=p.parse_args()
    if any(t<1 for t in a.times) or len(set(a.times))!=len(a.times):raise ValueError('Distinct positive time counts required')
    main(a)
