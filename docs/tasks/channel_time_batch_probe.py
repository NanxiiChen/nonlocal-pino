"""Finite production-grid timing/memory probe; no training labels or new input bank."""
import argparse
import importlib
import json
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

from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.temporal import sample_intervals,sample_continuous
from examples.channel_nsch2d.time_batching import make_accumulated_fn
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    baseline=Path(a.control)
    saved=json.loads((baseline/'config.json').read_text())
    c=importlib.import_module(a.config).get_config()
    from examples.channel_nsch2d.geometry import random_walls,initial_parameters
    random_geometry=random_walls(c)
    capacity_trial=a.capacity_trial or a.hardware_trial
    c.train.seed=saved['train']['seed']
    if not random_geometry:c.reference.root=saved['reference']['root']
    validate_config(c)
    temporal='time_derivative' in c
    causal=temporal and 'causal'in c
    balanced='balance'in c
    integral=c.nl.method=='integral_moment'
    from examples.channel_nsch2d.initialization import model_initialization_spec,load_initialized_model
    initialization=model_initialization_spec(c)
    # A warm physics-only mode names its source explicitly; cold modes keep
    # original raw state. Neither mode imports weights from this resource probe.
    now=json.loads(c.to_json());before=json.loads(json.dumps(saved))
    for item in (now,before):
        del item['log'];del item['train']['iterations'];del item['sample']['Nt']
    if 'microbatch' in now['train']:del now['train']['microbatch']
    if temporal:
        if a.hardware_trial:
            from examples.channel_nsch2d.capacity_validation import execution_config as comparison_config
        elif a.capacity_trial:
            from examples.channel_nsch2d.capacity_validation import comparison_config
        elif random_geometry:
            from examples.channel_nsch2d.random_wall_validation import comparison_config
        elif integral:
            from examples.channel_nsch2d.integral_validation import comparison_config
        elif balanced:
            from examples.channel_nsch2d.balance_validation import comparison_config
        else:
            from examples.channel_nsch2d.causal import comparison_config
        comparison_config(c,baseline)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU is occupied; do not overlap the timeJVP resource probe')
    elif now!=before:raise ValueError('Probe may change only Nt, microbatch and update budget')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    template=build_network(c,key)
    model=(template if capacity_trial else
        load_initialized_model(template,initialization)if initialization is not None else
        eqx.tree_deserialise_leaves(baseline/'model_initial.eqx',template))
    optimizer=make_optimizer(c)
    fresh=optimizer.init(eqx.filter(model,eqx.is_array))
    state=fresh if capacity_trial or initialization is not None else eqx.tree_deserialise_leaves(baseline/'optimizer_initial.eqx',fresh)
    if capacity_trial:
        eqx.tree_serialise_leaves(output/'model_initial.eqx',model)
        eqx.tree_serialise_leaves(output/'optimizer_initial.eqx',state)
    with np.load(baseline/'known_inputs.npz',allow_pickle=False) as f:
        if integral:
            from examples.channel_nsch2d.data import geometry_data
            parameters=(initial_parameters(c) if random_geometry else np.asarray(f['amplitudes'],dtype=np.float64))
            selected=parameters if capacity_trial else parameters[:c.operator.condition_batch]
            inputs=[geometry_data(a,c)[0]for a in selected]
            raw={k:np.stack([d[k]for d in inputs])for k in inputs[0]}
        else:raw={k:f[k][:c.operator.condition_batch]for k in f.files}
        bank={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer)
            else c.train.compute_dtype)for k,v in raw.items()}
        batch=jax.tree.map(lambda v:v[:c.operator.condition_batch],bank)
    loss=make_loss_fn(c)
    if causal:
        from examples.channel_nsch2d.causal import make_causal_fn,make_epsilon_update
        causal_objective=make_causal_fn(c);feedback=make_epsilon_update(c)
        epsilon=jnp.asarray(c.causal.adaptation.eps_init,dtype=c.train.compute_dtype)
        @eqx.filter_jit
        def causal_step(net,opt,batch,times,eps):
            value,parts,gradient,aux=causal_objective(net,batch,times,eps)
            nxt=feedback(aux['weights'],eps)
            finite=jnp.isfinite(value)&jnp.isfinite(nxt)&(nxt>0)&jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(a))for a in jax.tree.leaves((gradient,aux))if eqx.is_array(a)]))
            updates,opt=optimizer.update(gradient,opt,net)
            return eqx.apply_updates(net,updates),opt,value,parts,nxt,aux,finite
    elif balanced:
        from examples.channel_nsch2d.balance import make_balanced_accumulated_fn,group_names
        balanced_objective=make_balanced_accumulated_fn(c)
        weights=jnp.ones(len(group_names(c)),dtype=c.train.compute_dtype)
        @eqx.filter_jit
        def balanced_step(net,opt,batch,times,previous,refresh):
            value,parts,weights,norms,gradient=balanced_objective(net,batch,times,previous,refresh)
            finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))
                for a in jax.tree.leaves((gradient,weights,norms))if eqx.is_array(a)]))
            updates,opt=optimizer.update(gradient,opt,net)
            return eqx.apply_updates(net,updates),opt,value,parts,weights,norms,finite
    else:
        vg=make_accumulated_fn(c,loss) if 'microbatch' in c.train else eqx.filter_value_and_grad(loss,has_aux=True)
    if temporal:
        sample=jax.jit(lambda k:sample_continuous(k,c))
        _,data_key=jax.random.split(jax.random.PRNGKey(c.train.seed))
        def scheduled(step):
            _,kt=jax.random.split(jax.random.fold_in(data_key,step))
            return sample_continuous(kt,c)
        planned=np.asarray(jax.jit(jax.vmap(scheduled))(
            jnp.arange(c.train.iterations,dtype=jnp.uint32)))
        strata=np.arange(c.sample.Nt)*c.physical.terminal_time/c.sample.Nt
        if not np.all((planned>=strata)&(planned<strata+c.physical.terminal_time/c.sample.Nt)):
            raise ValueError('Planned actual GPU continuous-time draws violate strata')
        if not np.all(np.diff(planned,axis=-1)>0):raise ValueError('Planned time draws must be ordered')

    @eqx.filter_jit
    def step(net,opt,batch,times):
        (value,parts),gradient=vg(net,batch,times)
        if temporal:
            finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(g))
                for g in jax.tree.leaves(gradient)if eqx.is_array(g)]))
        updates,opt=optimizer.update(gradient,opt,net)
        if temporal:return eqx.apply_updates(net,updates),opt,value,parts,finite
        return eqx.apply_updates(net,updates),opt,value,parts

    timings=[];values=[];actual_times=[]
    for i in range(a.updates):
        if temporal:
            _,kt=jax.random.split(jax.random.fold_in(data_key,i));ts=sample(kt)
            actual_times.append(np.asarray(ts))
        else:_,ts=sample_intervals(jax.random.PRNGKey(i),c)
        start=time.perf_counter()
        if causal:
            model,state,value,parts,epsilon,aux,finite=causal_step(model,state,batch,ts,epsilon)
            if not bool(finite):raise FloatingPointError('Nonfinite actual timeJVP probe gradient/feedback')
        elif balanced:
            refresh=jnp.asarray(i%c.balance.update_every==0)
            model,state,value,parts,weights,norms,finite=balanced_step(model,state,batch,ts,weights,refresh)
            if not bool(finite):raise FloatingPointError('Nonfinite termwise full-time gradient or weights')
        elif temporal:
            model,state,value,parts,finite=step(model,state,batch,ts)
            if not bool(finite):raise FloatingPointError('Nonfinite actual uniform timeJVP probe gradient')
        else:model,state,value,parts=step(model,state,batch,ts)
        value.block_until_ready();timings.append(time.perf_counter()-start)
        if not np.isfinite(float(value)):raise FloatingPointError('Nonfinite probe loss')
        values.append(float(value))
        print(json.dumps(dict(update=i+1,seconds=timings[-1],loss=values[-1])),flush=True)
    physical_replay=None
    if initialization is not None:
        partial=dict(status='finite_gradient_updates_completed_query_gate_pending',updates=a.updates,
            all_update_seconds=timings,all_losses=values,memory=jax.devices()[0].memory_stats(),
            full_resource_probe_passed=False,fitted_probe_weights_exported=False)
        (output/'gradient-resource.json').write_text(json.dumps(partial,indent=2,allow_nan=False)+'\n')
        from docs.tasks.channel_projected_query_probe import check
        physical_replay=check(model,c,baseline)
    if capacity_trial:
        from examples.channel_nsch2d.network import encode,fields
        from examples.channel_nsch2d.data import geometry_data
        gate=json.loads((Path(c.reference.root)/'reference-validation.json').read_text())
        with np.load(gate['reference'],allow_pickle=False)as f:
            queries=jnp.asarray(f['points'],dtype=c.train.compute_dtype)
            query_times=jnp.asarray(f['time'],dtype=c.train.compute_dtype)
            raw_eval,_=geometry_data(f['amplitudes'],c)
        evaldata={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer)
            else c.train.compute_dtype)for k,v in raw_eval.items()}
        evaluate=eqx.filter_jit(lambda net:fields(net,encode(net,evaldata),queries,
            query_times,evaldata['amplitudes'],c))
        prediction=np.asarray(evaluate(model))
        if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite capacity-probe evaluation')
    result=dict(status='finite_production_probe_passed',config=json.loads(c.to_json()),
        control=str(baseline),updates=a.updates,time_samples=c.sample.Nt,
        microbatch=c.train.microbatch.to_dict() if 'microbatch' in c.train else None,
        cold_first_update_seconds=timings[0],warm_update_seconds=float(np.mean(timings[1:])),
        all_update_seconds=timings,all_losses=values,memory=jax.devices()[0].memory_stats(),
        device=jax.devices()[0].device_kind,
        scope=(f'Resource probe on B{c.operator.condition_batch} fresh initial-bank inputs'
            if random_geometry else 'Resource probe on fixed original inputs')+'; not an accuracy/generalization experiment',
        fitted_probe_weights_exported=False,
        evolved_training_labels=0,FEM_solves=0,optimizer_updates_per_time_batch=1)
    if capacity_trial:
        result['capacity_trial']=dict(full_geometry_pool_on_device=True,
            geometry_pool_size=len(parameters),finite_nominal_evaluation=True,
            cold_state_saved_before_probe_updates=True,
            parameter_count=sum(a.size for a in jax.tree.leaves(model)if eqx.is_array(a)))
    if a.hardware_trial:
        import csv
        gpu=list(csv.reader(subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()))
        if len(gpu)!=1:raise ValueError('Hardware timing requires one visible physical GPU')
        result['hardware_trial']=dict(gpu_name=gpu[0][0].strip(),gpu_uuid=gpu[0][1].strip(),
            total_memory_bytes=int(gpu[0][2])*2**20,time_microbatch=c.train.microbatch.size,
            scope='Same effective condition/time objective; changed hardware and explicit execution block only')
    if initialization is not None:
        result['model_initialization']=initialization
        result['physical_query_replay']=physical_replay
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    if temporal:
        np.testing.assert_array_equal(np.asarray(actual_times),planned[:a.updates])
        if causal:
            result['adaptive_feedback']=dict(final_next_epsilon=float(epsilon),
                final_mean_weights=np.asarray(aux['weights']).mean(axis=-1).tolist(),
                exact_time_jvp=True,space=c.nl.method,extra_spatial_residual_AD=False)
        elif balanced:
            result['termwise_balance']=dict(settings=c.balance.to_dict(),groups=group_names(c),
                all_gradient_and_weight_updates_finite=True,final_weights=np.asarray(weights).tolist(),
                refresh_updates=[i+1 for i in range(a.updates)if i%c.balance.update_every==0],
                warm_refresh_seconds=[timings[i]for i in range(1,a.updates)if i%c.balance.update_every==0],
                warm_held_seconds=float(np.mean([timings[i]for i in range(1,a.updates)if i%c.balance.update_every!=0])),
                scope='Norms after complete40-time per-term parameter-gradient accumulation; fitted probe state discarded')
        else:
            result['uniform_time_objective']=dict(causal_weights=False,all_gradient_updates_finite=True,
                exact_time_jvp=True,space=c.nl.method,extra_spatial_residual_AD=False)
        result['time_sampler']=dict(planned_updates=c.train.iterations,
            all_planned_GPU_strata_and_order_passed=True,actual_probe_times_exact_global_key_replay=True,
            repeated_time_batch=bool(np.array_equal(actual_times[0],actual_times[1])))
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    root=Path(__file__).resolve().parents[2]
    relative=Path(a.config.replace('.','/')+'.py');target=output/'source'/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/relative,target)
    for name in ('docs/tasks/channel_time_batch_probe.py','examples/channel_nsch2d/time_batching.py',
                 'examples/channel_nsch2d/train.py','examples/channel_nsch2d/problem.py'):
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    if temporal:
        for name in ('examples/channel_nsch2d/network.py','examples/channel_nsch2d/losses.py',
                     'examples/channel_nsch2d/causal.py','examples/channel_nsch2d/temporal.py',
                     'src/nonlocal_pinn/training/causal.py','src/nonlocal_pinn/training/optimizers.py'):
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/name,target)
    if initialization is not None:
        names=('initialization.py','chemical_projection.py','projected_fields.py',
            'configs/train_operator_width128_gmls_time40_projected_mu_refine_online.py')
        for name in names:
            relative=Path('examples/channel_nsch2d')/name;target=output/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/relative,target)
        relative=Path('docs/tasks/channel_projected_query_probe.py');target=output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/relative,target)
    if balanced:
        for name in ('balance.py','balance_validation.py'):
            relative=Path('examples/channel_nsch2d')/name;target=output/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/relative,target)
    if random_geometry:
        shutil.copytree(root/'examples/channel_nsch2d',output/'source/examples/channel_nsch2d',
            dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','control','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--updates',type=int,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--capacity-trial',action='store_true',help='Explicit random-wall192/B8/10k cold resource trial')
    mode.add_argument('--hardware-trial',action='store_true',help='Same effective192/B8/10k objective, timeblock2or8 hardware timing')
    args=p.parse_args()
    if args.updates<3:raise ValueError('Require a cold update and at least two warm updates')
    main(args)
