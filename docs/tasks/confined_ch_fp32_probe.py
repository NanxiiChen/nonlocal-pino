"""Same-device full-bank FP32 probe and independent CPU field replay."""
import argparse
import json
from pathlib import Path
import re
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.training.runtime import configure_gpu_policy,with_run_seed
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width128_fp32 import get_config
from examples.confined_ch2d.operator_training import build_network,load_bank,make_loss,operator_temporal_samples,fields
from examples.confined_ch2d.operator_precision import make_compute_optimizer


def check32(tree):
    for a in jax.tree.leaves(tree):
        if eqx.is_inexact_array(a):
            if a.dtype!=jnp.float32 or not np.isfinite(np.asarray(a)).all():
                raise FloatingPointError(f'Expected finite FP32 leaf: {a.shape}/{a.dtype}')


def probe(prior,output):
    output.mkdir(parents=True,exist_ok=False)
    c=with_run_seed(get_config(),None)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    if jax.default_backend()!='gpu':raise RuntimeError('Probe requires GPU')
    old=ConfigDict(json.loads((prior/'config.json').read_text()))
    expected=json.loads(c.to_json());expected['train']['compute_dtype']='float64';expected['log']['run_name']=old.log.run_name
    assert expected==json.loads(old.to_json())
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    started=time.perf_counter()
    print('Loading unchanged full520geometry bank; final compute arrays FP32',flush=True)
    _,_,bank=load_bank(c)
    assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    check32(bank)
    keys=jax.random.split(jax.random.key(c.train.seed),3)
    net=build_network(c,keys[0])
    saved=eqx.tree_deserialise_leaves(prior/'model_initial.eqx',build_network(old,jax.random.key(0)))
    for a,b in zip(jax.tree.leaves(net),jax.tree.leaves(saved),strict=True):
        if eqx.is_array(a):np.testing.assert_array_equal(a,np.asarray(b).astype(np.float32))
    del saved
    opt=make_compute_optimizer(c);state=opt.init(eqx.filter(net,eqx.is_array));loss=make_loss(c)
    @eqx.filter_jit
    def step(model,state,batch,times):
        value,g=eqx.filter_value_and_grad(loss)(model,batch,times)
        finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(g) if eqx.is_array(a)]))
        delta,state=opt.update(g,state,model)
        return eqx.apply_updates(model,delta),state,value,finite
    ids=jax.random.randint(jax.random.fold_in(keys[1],1),(c.operator.condition_batch,),0,c.operator.train_conditions)
    batch=jax.tree.map(lambda a:a[ids],bank)
    times,_=operator_temporal_samples(c,1,jax.random.fold_in(keys[2],1))
    lower=step.lower(net,state,batch,times);ir=lower.as_text()
    (output/'step.stablehlo').write_text(ir)
    f64=re.findall(r'tensor<[^>]*xf64>',ir)
    if f64:raise AssertionError(f'Non-scalar FP64 tensors remain in FP32 training IR: {sorted(set(f64))}')
    scalar64=len(re.findall(r'tensor<f64>',ir))
    tick=time.perf_counter();compiled=lower.compile();compile_seconds=time.perf_counter()-tick
    losses=[];durations=[];draws=[]
    for iteration in range(1,26):
        ids=jax.random.randint(jax.random.fold_in(keys[1],iteration),(c.operator.condition_batch,),0,c.operator.train_conditions)
        batch=jax.tree.map(lambda a:a[ids],bank)
        key=jax.random.fold_in(keys[2],iteration)
        times,_=operator_temporal_samples(c,iteration,key);old_times,_=operator_temporal_samples(old,iteration,key)
        np.testing.assert_array_equal(times,np.asarray(old_times).astype(np.float32))
        tick=time.perf_counter();net,state,value,finite=compiled(net,state,batch,times)
        value.block_until_ready();durations.append(time.perf_counter()-tick)
        if not bool(finite):raise FloatingPointError(f'Probe nonfinite gradient at {iteration}')
        check32((net,state));losses.append(float(value));draws.append(np.asarray(ids))
        print(json.dumps(dict(step=iteration,seconds=durations[-1],loss=losses[-1])),flush=True)
    # Compare actual geometry prefix to the completed width96 control.
    prior96=prior.parent/'20261030-confined-geometry-b16-t16-wide96-soap-01'
    with np.load(prior96/'training_draws.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(draws,f['condition_ids'][:25])
    index=512;data=jax.tree.map(lambda a:a[index],bank)
    t=jnp.asarray([0.,.75,1.5,2.25,3.],dtype=jnp.float32)
    prediction=np.asarray(eqx.filter_jit(lambda n,d:fields(n,d,t,c))(net,data))
    check32(prediction)
    eqx.tree_serialise_leaves(output/'model_probe.eqx',net)
    np.savez_compressed(output/'cpu_replay.npz',**{k:np.asarray(v) for k,v in data.items()},times=np.asarray(t),prediction=prediction)
    memory=jax.devices()[0].memory_stats()
    result=dict(status='passed',prior_fp64=str(prior),config=c.to_dict(),steps=25,
        initial_weights_equal_rounded_fp64=True,time_draws_equal_rounded_fp64=True,
        actual_condition_prefix_matches_completed_control=True,all_compute_leaves_fp32=True,
        stablehlo_nonscalar_fp64_tensors=0,stablehlo_fp64_scalar_tokens=scalar64,
        all_losses_gradients_states_finite=True,
        compile_seconds=compile_seconds,first_execution_seconds=durations[0],
        warm_seconds=float(np.mean(durations[5:])),durations=durations,losses=losses,
        memory=memory,wall_seconds=time.perf_counter()-started,
        scope='25discarded feasibility updates on existing GPU; subsequent training restarts cold. Warm20steps cover fourSOAP refreshes. Highest matmul precision, no mixed fallback or claimed accuracy gain.')
    (output/'probe.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if memory['peak_bytes_in_use']>.95*memory['bytes_limit']:raise RuntimeError('Insufficient allocator headroom')
    print(json.dumps({k:v for k,v in result.items() if k not in ['config','durations','losses']},indent=2),flush=True)


def cpu_replay(output):
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    if jax.default_backend()!='cpu':raise RuntimeError('Replay must use CPU')
    c=ConfigDict(json.loads((output/'config.json').read_text()))
    model=eqx.tree_deserialise_leaves(output/'model_probe.eqx',build_network(c,jax.random.key(0)))
    with np.load(output/'cpu_replay.npz',allow_pickle=False) as f:
        times=f['times'];expected=f['prediction'];data={k:jnp.asarray(f[k]) for k in f.files if k not in ['times','prediction']}
    actual=np.asarray(eqx.filter_jit(lambda n,d:fields(n,d,jnp.asarray(times),c))(model,data))
    gaps=np.max(abs(actual-expected),axis=(0,1));tolerances=np.array([2e-5,2e-4])
    if not np.all(gaps<tolerances):raise AssertionError((gaps,tolerances))
    result=dict(status='passed',field_maximum_differences=gaps.tolist(),field_tolerances=tolerances.tolist(),scope='Independent CPU replay of25stepGPU probe; not final training accuracy')
    (output/'cpu_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def initial_audit(prior,run,output):
    if output.exists():raise FileExistsError(output)
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise RuntimeError('Initial audit must use CPU')
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    old=ConfigDict(json.loads((prior/'config.json').read_text()))
    expected=json.loads(c.to_json());expected['train']['compute_dtype']='float64';expected['log']['run_name']=old.log.run_name
    assert expected==json.loads(old.to_json())
    new=eqx.tree_deserialise_leaves(run/'model_initial.eqx',build_network(c,jax.random.key(0)))
    previous=eqx.tree_deserialise_leaves(prior/'model_initial.eqx',build_network(old,jax.random.key(0)))
    count=0
    for a,b in zip(jax.tree.leaves(new),jax.tree.leaves(previous),strict=True):
        if eqx.is_array(a):
            np.testing.assert_array_equal(a,np.asarray(b).astype(np.float32));count+=1
    for source in ['src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py']:
        assert (run/'source'/source).read_bytes()==(prior/'source'/source).read_bytes(),source
    metrics=[json.loads((p/'metrics.jsonl').read_text().splitlines()[0]) for p in [prior,run]]
    for name in ['test_phi_pooled','test_phi_terminal','test_mu_pooled']:
        assert abs(metrics[0][name]-metrics[1][name])<2e-4,(name,metrics)
    result=dict(status='passed',run=str(run),prior=str(prior),initial_array_leaves=count,
        all_initial_arrays_equal_exact_rounded_fp64=True,only_config_change='train.compute_dtype float64->float32 and runtime log name',
        network_optimizer_runtime_sources_unchanged=True,
        scope='Same fresh initialization rounded toFP32, not continuation. Old interrupted actual draw trace unavailable; new completed trace independently checked against globalkeys and completed width96prefix.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('probe','cpu','initial'),required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.action=='probe':probe(a.prior.resolve(),a.output.resolve())
    elif a.action=='cpu':cpu_replay(a.output.resolve())
    else:
        if a.run is None:p.error('--run is required for initial audit')
        initial_audit(a.prior.resolve(),a.run.resolve(),a.output.resolve())
