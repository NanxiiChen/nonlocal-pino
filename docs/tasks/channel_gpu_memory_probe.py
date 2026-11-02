"""Compile memory preflight and finite full-gradient timings on the96GB GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def main(a):
    from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8 import get_config
    from examples.channel_nsch2d.capacity_validation import comparison_config
    from examples.channel_nsch2d.data import build_bank,geometry_data
    from examples.channel_nsch2d.losses import make_loss_fn
    from examples.channel_nsch2d.network import build_network,encode,fields
    from examples.channel_nsch2d.problem import validate_config
    from examples.channel_nsch2d.temporal import sample_continuous
    from examples.channel_nsch2d.time_batching import make_accumulated_fn
    from nonlocal_pinn.training.optimizers import make_optimizer
    from nonlocal_pinn.training.runtime import configure_gpu_policy

    if (a.time_block not in (2,4,5,8,10,20,40) or a.updates<11
            or not 0<a.memory_fraction<=.97):
        raise ValueError('Require a40-time divisor, >=11updates and explicit allocator fraction in(0,.97]')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; do not overlap memory probes')
    gpu=list(csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,uuid,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()))
    if len(gpu)!=1:raise ValueError('Exactly one physical GPU required')
    physical_bytes=int(gpu[0][2])*2**20
    control=Path(a.control);c=get_config()
    c.train.seed=json.loads((control/'config.json').read_text())['train']['seed']
    comparison_config(c,control)
    c.train.microbatch.size=a.time_block;validate_config(c)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    shutil.copytree(ROOT/'examples/channel_nsch2d',output/'source/examples/channel_nsch2d',
        ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('docs/tasks/channel_gpu_memory_probe.py','src/nonlocal_pinn/training/optimizers.py',
                     'src/nonlocal_pinn/training/runtime.py','src/nonlocal_pinn/nn/point_conditional.py'):
        path=output/'source'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,path)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(a.memory_fraction)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    model_key,data_key=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=build_network(c,model_key);optimizer=make_optimizer(c)
    state=optimizer.init(eqx.filter(model,eqx.is_array))
    eqx.tree_serialise_leaves(output/'model_initial.eqx',model)
    eqx.tree_serialise_leaves(output/'optimizer_initial.eqx',state)
    raw,_=build_bank(c)
    convert=lambda data:{k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer)
        else c.train.compute_dtype)for k,v in data.items()}
    bank=convert(raw);batch=jax.tree.map(lambda v:v[:c.operator.condition_batch],bank)
    objective=make_accumulated_fn(c,make_loss_fn(c))

    @eqx.filter_jit
    def step(net,opt,batch,times):
        (value,parts),gradient=objective(net,batch,times)
        finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(g))
            for g in jax.tree.leaves(gradient)if eqx.is_array(g)]))
        updates,opt=optimizer.update(gradient,opt,net)
        return eqx.apply_updates(net,updates),opt,value,parts,finite

    sample=jax.jit(lambda i:sample_continuous(jax.random.split(jax.random.fold_in(data_key,i))[1],c))
    first=sample(jnp.asarray(0,dtype=jnp.uint32))
    begin=time.perf_counter();compiled=step.lower(model,state,batch,first).compile()
    compile_seconds=time.perf_counter()-begin
    analysis=compiled.compiled.memory_analysis()
    if analysis is None:raise RuntimeError('GPU compiler did not expose memory analysis')
    stats=jax.devices()[0].memory_stats()
    memory_analysis=dict(argument_bytes=analysis.argument_size_in_bytes,output_bytes=analysis.output_size_in_bytes,
        alias_bytes=analysis.alias_size_in_bytes,temporary_bytes=analysis.temp_size_in_bytes,
        resident_bytes=stats['bytes_in_use'])
    required=stats['bytes_in_use']+analysis.output_size_in_bytes+analysis.temp_size_in_bytes-analysis.alias_size_in_bytes
    limit=min(.95*physical_bytes,.97*stats['bytes_limit'])
    memory_analysis.update(estimated_peak_bytes=required,allowed_peak_bytes=int(limit),compile_seconds=compile_seconds)
    (output/'memory-analysis.json').write_text(json.dumps(memory_analysis,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='compiled_memory_preflight',**memory_analysis)),flush=True)
    hardware=dict(gpu_name=gpu[0][0].strip(),gpu_uuid=gpu[0][1].strip(),total_memory_bytes=physical_bytes,
        time_microbatch=a.time_block,allocator_fraction=a.memory_fraction)
    common=dict(config=json.loads(c.to_json()),control=str(control),hardware_trial=hardware,
        memory_preflight=memory_analysis,fitted_probe_weights_exported=False,evolved_training_labels=0,FEM_solves=0)
    if required>limit:
        result=dict(status='skipped_memory_preflight',optimizer_updates=0,**common)
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return
    timings=[];values=[];draws=[]
    for i in range(a.updates):
        ts=sample(jnp.asarray(i,dtype=jnp.uint32));draws.append(np.asarray(ts))
        begin=time.perf_counter()
        model,state,value,parts,finite=compiled(model,state,batch,ts)
        jax.block_until_ready((model,state,value,parts,finite))
        timings.append(time.perf_counter()-begin)
        if not bool(finite):raise FloatingPointError('Nonfinite complete aggressive-memory gradient')
        values.append(float(value))
        print(json.dumps(dict(update=i+1,seconds=timings[-1],loss=values[-1])),flush=True)
    gate=json.loads((Path(c.reference.root)/'reference-validation.json').read_text())
    with np.load(gate['reference'],allow_pickle=False)as f:
        points=jnp.asarray(f['points'],dtype=c.train.compute_dtype)
        times=jnp.asarray(f['time'],dtype=c.train.compute_dtype)
        raw_eval,_=geometry_data(f['amplitudes'],c)
    evaluation=convert(raw_eval)
    evaluate=eqx.filter_jit(lambda net:fields(net,encode(net,evaluation),points,times,evaluation['amplitudes'],c))
    prediction=np.asarray(evaluate(model))
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite nominal evaluation')
    strata=np.arange(c.sample.Nt)*c.physical.terminal_time/c.sample.Nt;draws=np.asarray(draws)
    if not np.all((draws>=strata)&(draws<strata+c.physical.terminal_time/c.sample.Nt)):
        raise ValueError('Actual probe time draws violate the unchanged strata')
    result=dict(status='finite_production_probe_passed',updates=a.updates,time_samples=c.sample.Nt,
        microbatch=c.train.microbatch.to_dict(),warm_update_seconds=float(np.mean(timings[1:])),
        cold_first_update_seconds=compile_seconds+timings[0],all_update_seconds=timings,all_losses=values,
        memory=jax.devices()[0].memory_stats(),device=jax.devices()[0].device_kind,
        capacity_trial=dict(full_geometry_pool_on_device=True,geometry_pool_size=c.operator.train_conditions,
            finite_nominal_evaluation=True,cold_state_saved_before_probe_updates=True,
            parameter_count=sum(v.size for v in jax.tree.leaves(model)if eqx.is_array(v))),
        optimizer_updates_per_time_batch=1,scope='Finite resource timing, same B8/Nt40 objective; no accuracy claim',**common)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--time-block',type=int,required=True)
    p.add_argument('--memory-fraction',type=float,required=True)
    p.add_argument('--updates',type=int,required=True)
    main(p.parse_args())
