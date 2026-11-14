"""Canonical physics-only channel training, cold or explicitly continued."""
from pathlib import Path
from contextlib import nullcontext
import sys
if __package__ is None or __package__=='':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from absl import app,flags
from ml_collections import config_flags
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import json
import shutil
import time
from datetime import datetime,timezone
from examples.channel_nsch2d.problem import validate_config,identity
from examples.channel_nsch2d.data import build_bank,geometry_data
from examples.channel_nsch2d.network import build_network,encode,fields,phase_jacobian
from examples.channel_nsch2d.losses import make_loss_fn,part_names
from examples.channel_nsch2d.balance import group_names,make_balanced_fn,make_balanced_accumulated_fn
from examples.channel_nsch2d.partition import stage_at,mask_output_updates
from examples.channel_nsch2d.phase_coordinate import jacobian_diagnostics
from examples.channel_nsch2d.temporal import sample_intervals
from examples.channel_nsch2d.continuation import continuation_spec,restore_state
from examples.channel_nsch2d.online import RollingGeometryPool,replace_slots
from ml_collections import ConfigDict
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy,with_run_seed

_CONFIG=config_flags.DEFINE_config_file('config',str(Path(__file__).parent/'configs/train_default.py'))
_SEED=flags.DEFINE_integer('seed',None,'Run provenance, not a strategy parameter')


def train(c):
    entry_start=time.perf_counter()
    validate_config(c)
    if 'refinement' in c.train:
        from examples.channel_nsch2d.refinement import train_refinement
        return train_refinement(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    continuation=continuation_spec(c)
    from examples.channel_nsch2d.initialization import model_initialization_spec,load_initialized_model
    initialization=model_initialization_spec(c)
    root=Path(c.reference.root)
    gate=json.loads((root/'reference-validation.json').read_text())
    if gate['status']!='passed':raise ValueError('Independent FEM refinement must pass before training')
    if json.loads(json.dumps(identity(c)))!=gate['identity']:raise ValueError('Reference physical identity mismatch')
    with np.load(gate['reference'],allow_pickle=False) as file:
        reference={k:file[k] for k in file.files}
    if not np.array_equal(reference['time'],np.linspace(0,c.physical.terminal_time,c.reference.snapshots)):
        raise ValueError('Reference time mismatch')
    if 'phase_coordinate' in c:
        if np.max(np.abs(reference['fields'][...,0]))>=c.phase_coordinate.amplitude:
            raise ValueError('Phase output range excludes the saved development FEM; increase explicit map amplitude')
    dtype=np.dtype(c.train.compute_dtype)
    online='online' in c.operator
    forbidden=reference['amplitudes'][None,:]
    if 'validation_manifest' in c.operator:
        from examples.channel_nsch2d.cohort import reserved_amplitudes
        forbidden=np.concatenate((forbidden,reserved_amplitudes(c)))
    convert=lambda d:{k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else dtype)
                       for k,v in d.items()}
    bank,operator_reports=build_bank(c);actual_bank=bank;bank=convert(bank)
    evaldata,_=geometry_data(reference['amplitudes'],c);evaldata=convert(evaldata)
    evalpoints=jnp.asarray(reference['points'],dtype=dtype)
    evaltimes=jnp.asarray(reference['time'],dtype=dtype)
    model_key,data_key=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=load_initialized_model(build_network(c,model_key),initialization);optimizer=make_optimizer(c)
    net,state,start_step=restore_state(net,optimizer,continuation);loss=make_loss_fn(c)
    adaptive='balance' in c
    staggered='stagger' in c
    causal='causal' in c
    feedback_causal=causal and 'adaptation' in c.causal
    adaptive_fn=make_balanced_fn(c) if adaptive else None
    if adaptive and 'microbatch' in c.train:adaptive_fn=make_balanced_accumulated_fn(c)
    accumulated_fn=None
    if 'microbatch' in c.train and not causal and not adaptive:
        from examples.channel_nsch2d.time_batching import make_accumulated_fn
        accumulated_fn=make_accumulated_fn(c,loss)
    adaptive_weights=jnp.ones(len(group_names(c)),dtype=dtype)
    @eqx.filter_jit
    def step(model,optim_state,batch,times,weights,refresh,active):
        if accumulated_fn is not None:
            (total,parts),gradient=accumulated_fn(model,batch,times)
            norms=jnp.zeros_like(weights)
        elif adaptive:
            total,parts,weights,norms,gradient=adaptive_fn(model,batch,times,weights,refresh)
        elif staggered:
            (total,parts),gradient=eqx.filter_value_and_grad(loss,has_aux=True)(model,batch,times,active)
            norms=jnp.zeros_like(weights)
        else:
            (total,parts),gradient=eqx.filter_value_and_grad(loss,has_aux=True)(model,batch,times)
            norms=jnp.zeros_like(weights)
        updates,optim_state=optimizer.update(gradient,optim_state,model)
        if staggered:updates=mask_output_updates(updates,active)
        return eqx.apply_updates(model,updates),optim_state,total,parts,weights,norms
    if causal:
        from examples.channel_nsch2d.causal import make_causal_fn,epsilon_at,make_epsilon_update
        causal_fn=make_causal_fn(c)
        epsilon_update=make_epsilon_update(c) if feedback_causal else None
        if feedback_causal:epsilon=jnp.asarray(c.causal.adaptation.eps_init,dtype=dtype)
        @eqx.filter_jit
        def causal_step(model,optim_state,batch,times,epsilon):
            total,parts,gradient,aux=causal_fn(model,batch,times,epsilon)
            if feedback_causal:
                aux['next_epsilon']=epsilon_update(aux['weights'],epsilon)
                aux['mean_weights']=jnp.mean(aux['weights'],axis=-1)
            finite=jnp.isfinite(total)&jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves((gradient,aux)) if eqx.is_array(a)]))
            if feedback_causal:finite=finite&(epsilon>0)&(aux['next_epsilon']>0)
            updates,optim_state=optimizer.update(gradient,optim_state,model)
            return eqx.apply_updates(model,updates),optim_state,total,parts,aux,finite
    query_record=None
    if 'chemical_constraint'in c:
        from examples.channel_nsch2d.projected_fields import make_query_fn,query_rule
        query_record=query_rule({k:np.asarray(v)for k,v in evaldata.items()},np.asarray(evalpoints),c)
        query_device={k:jnp.asarray(v)for k,v in query_record.items()};query_value=make_query_fn(c)
        evaluate=eqx.filter_jit(lambda model:query_value(model,encode(model,evaldata),evaldata,evalpoints,evaltimes,query_device))
    else:
        evaluate=eqx.filter_jit(lambda model:fields(model,encode(model,evaldata),evalpoints,
            evaltimes,evaldata['amplitudes'],c))
    if 'phase_coordinate' in c:
        evaluate_jacobian=eqx.filter_jit(lambda model:phase_jacobian(model,
            encode(model,evaldata),evalpoints,evaltimes,c))
    name=c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-channel-gmls'
    run=Path(c.log.log_dir)/name;run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(c.to_json(indent=2)+'\n')
    (run/'operators.json').write_text(json.dumps(operator_reports,indent=2)+'\n')
    np.savez_compressed(run/'known_inputs.npz',**actual_bank)
    if online:
        np.savez_compressed(run/'evaluation_inputs.npz',**{k:np.asarray(v) for k,v in evaldata.items()})
        actual_amplitudes=actual_bank['amplitudes'].astype(dtype).copy()
        if np.any(np.all(actual_amplitudes[:,None,:]==forbidden.astype(dtype)[None,:,:],axis=-1)):
            raise ValueError('Initial training bank contains a reserved evaluation geometry')
        np.savez_compressed(run/'reserved_geometries.npz',amplitudes=forbidden)
    shutil.copytree(Path(__file__).parent,run/'source/examples/channel_nsch2d',ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/training/optimizers.py',
                     'src/nonlocal_pinn/training/runtime.py','src/nonlocal_pinn/training/spatial_sampling.py'):
        target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[2]/relative,target)
    if causal:
        relative='src/nonlocal_pinn/training/causal.py';target=run/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[2]/relative,target)
    eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    eqx.tree_serialise_leaves(run/'optimizer_initial.eqx',state)
    if initialization is not None:
        if (run/'model_initial.eqx').read_bytes()!=Path(initialization['model_path']).read_bytes():
            raise ValueError('Imported physics model bytes differ from source')
        (run/'initialization.json').write_text(json.dumps(initialization,indent=2,allow_nan=False)+'\n')
    if query_record is not None:np.savez_compressed(run/'evaluation_query_rule.npz',**query_record)
    if continuation is not None:
        parent_c=ConfigDict(continuation['parent_config'])
        with np.load(continuation['prediction_path'],allow_pickle=False) as file:parent_saved={k:file[k] for k in file.files}
        parent_data,_=geometry_data(parent_saved['amplitudes'],parent_c);parent_data=convert(parent_data)
        parent_replay=eqx.filter_jit(lambda model:fields(model,encode(model,parent_data),
            jnp.asarray(parent_saved['points'],dtype=dtype),jnp.asarray(parent_saved['time'],dtype=dtype),
            parent_data['amplitudes'],parent_c))
        replay=np.asarray(parent_replay(net),dtype=np.float64)
        replay_gap=float(np.max(np.abs(replay-parent_saved['fields'])))
        if replay_gap>2e-5 or not np.isfinite(replay).all():raise ValueError(f'Parent physical replay failed: {replay_gap}')
        if (run/'model_initial.eqx').read_bytes()!=Path(continuation['model_path']).read_bytes():raise ValueError('Imported model bytes differ')
        if (run/'optimizer_initial.eqx').read_bytes()!=Path(continuation['optimizer_path']).read_bytes():raise ValueError('Imported complete optimizer bytes differ')
        changed=np.asarray(evaluate(net),dtype=np.float64)
        continuation.update(parent_replay_max_gap=replay_gap,
            refined_conditioning_field_max_change=float(np.max(np.abs(changed-replay))),
            initial_model_optimizer_bytes_exact=True)
        (run/'continuation.json').write_text(json.dumps(continuation,indent=2,allow_nan=False)+'\n')
    updates=c.train.iterations-start_step
    sampled=np.empty((updates,c.operator.condition_batch),dtype=np.int32)
    discrete='time_discretization' in c
    spatial_study='spatial' in c.sample
    fem_study = c.nl.method == 'fem_weak_p2p1'
    compact=spatial_study and c.sample.spatial.mode!='full'
    spatial_pool=nullcontext(None)
    if compact:
        from examples.channel_nsch2d.spatial_sampling import SpatialPool
        spatial_pool=SpatialPool(c,actual_bank)
        core_ids=np.empty((updates,c.operator.condition_batch,c.sample.spatial.core_count),dtype=np.int32)
        pool_ids=np.empty_like(sampled)
        halo_counts=np.empty_like(sampled)
    spatial_seconds=[]
    continuous_sampler=None
    if 'time_derivative' in c:
        from examples.channel_nsch2d.temporal import sample_continuous
        continuous_sampler=jax.jit(lambda key:sample_continuous(key,c))
    sampled_times=np.empty((updates,c.sample.Nt),dtype=dtype if discrete else np.float32)
    if online:
        sampled_draw_ids=np.empty((updates,c.operator.condition_batch),dtype=np.int64)
        from examples.channel_nsch2d.geometry import parameter_count
        sampled_amplitudes=np.empty((updates,c.operator.condition_batch,parameter_count(c)),dtype=dtype)
        replace=jax.jit(replace_slots,donate_argnums=(0,))
    if discrete:sampled_intervals=np.empty((updates,c.sample.Nt),dtype=np.int32)
    timings=[];start=time.perf_counter();prediction=None
    stage_counts=dict(phase=0,flow=0,joint=0)
    def assess(completed):
        nonlocal prediction
        prediction=np.asarray(evaluate(net),dtype=np.float64)
        if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite physical prediction')
        target=reference['fields']
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        metrics=dict(step=completed,elapsed=time.perf_counter()-start,
            phi=relative(prediction[...,0],target[...,0]),terminal_phi=relative(prediction[-1,:,0],target[-1,:,0]),
            dispersed_phase=relative((1-prediction[...,0])/2,(1-target[...,0])/2),
            velocity=relative(prediction[1:,:,2:4],target[1:,:,2:4]),
            pressure=relative(prediction[1:,:,4],target[1:,:,4]),
            chemical=relative(prediction[1:,:,1],target[1:,:,1]),
            phi_range=[float(prediction[...,0].min()),float(prediction[...,0].max())],
            reference=str(gate['reference']),scope='One local-FEM geometry; not a generalization estimate')
        if 'phase_coordinate' in c:
            jacobian=np.asarray(evaluate_jacobian(net))
            if not np.isfinite(jacobian).all():raise FloatingPointError('Nonfinite phase output Jacobian')
            metrics['phase_map']=dict(amplitude=c.phase_coordinate.amplitude,
                **jacobian_diagnostics(jacobian,c.phase_coordinate.amplitude),
                near_bound_fraction=float(np.mean(np.abs(prediction[...,0])>.99*c.phase_coordinate.amplitude)),
                terminal_negative_queries=int(np.sum(prediction[-1,:,0]<0)))
        with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        print(json.dumps(metrics,allow_nan=False),flush=True)
        return metrics
    metrics=assess(start_step)
    if 'phase_coordinate' in c:
        np.savez_compressed(run/'prediction_initial.npz',fields=prediction)
    producer=(RollingGeometryPool(c,run,forbidden) if online else nullcontext(None))
    with (run/'loss.jsonl').open('x') as stream,producer as pool,spatial_pool as spatial:
        for i in range(start_step,c.train.iterations):
            offset=i-start_step
            if online:
                replacement=pool.before_step(i)
                if replacement is not None:
                    slots,values=replacement
                    bank=replace(bank,jnp.asarray(slots),convert(values))
                    actual_amplitudes[slots]=values['amplitudes']
                    if compact:spatial.install(slots,values,pool.draw_ids[slots],i)
            ki=jax.random.fold_in(data_key,i);kg,kt=jax.random.split(ki)
            ids=jax.random.randint(kg,(c.operator.condition_batch,),0,c.operator.train_conditions)
            # Uniform stratified physical times. Optional zero is an explicit
            # control; the current compact-IC recipe trains only positive times.
            if discrete:
                intervals,times=sample_intervals(kt,c)
                sampled_intervals[offset]=np.asarray(intervals)
            elif continuous_sampler is not None:
                times=continuous_sampler(kt)
            else:
                draws=jax.random.uniform(kt,(c.sample.Nt,),dtype=jnp.float32)
                times=(jnp.arange(c.sample.Nt)+draws)*c.physical.terminal_time/c.sample.Nt
            sampled[offset]=np.asarray(ids);sampled_times[offset]=np.asarray(times)
            if online:
                sampled_draw_ids[offset]=pool.draw_ids[sampled[offset]]
                sampled_amplitudes[offset]=actual_amplitudes[sampled[offset]]
            if c.sample.include_initial_time:times=jnp.concatenate((jnp.zeros(1,dtype=times.dtype),times))
            times=times.astype(dtype)
            prepare_start=time.perf_counter()
            if compact:
                raw,choices,cores,counts=spatial.take(sampled[offset],i)
                pool_ids[offset],core_ids[offset],halo_counts[offset]=choices,cores,counts
                batch=convert(raw)
            else:
                batch=jax.tree.map(lambda a:a[ids],bank)
            if spatial_study or fem_study:
                jax.block_until_ready(batch)
                spatial_seconds.append(time.perf_counter()-prepare_start)
            refresh=adaptive and i%c.balance.update_every==0
            stage,active=stage_at(i,c.stagger) if staggered else ('joint',(True,True))
            stage_counts[stage]+=1
            if causal and not feedback_causal:epsilon=jnp.asarray(epsilon_at(i,c),dtype=dtype)
            begin=time.perf_counter()
            if causal:
                epsilon_used=epsilon
                net,state,total,parts,causal_aux,finite=causal_step(net,state,batch,times,epsilon)
                if not bool(finite):raise FloatingPointError(f'Nonfinite causal gradient or diagnostics at update{i+1}')
            else:
                net,state,total,parts,adaptive_weights,norms=step(net,state,batch,times,
                    adaptive_weights,jnp.asarray(refresh),jnp.asarray(active,dtype=dtype))
            if spatial_study or fem_study:
                jax.block_until_ready((net,state,total))
            else:
                total.block_until_ready()
            timings.append(time.perf_counter()-begin)
            if (spatial_study or fem_study) and offset==24:
                leaves=[a for a in jax.tree.leaves((net,state)) if eqx.is_inexact_array(a)]
                if not all(np.isfinite(a).all() for a in leaves):
                    raise FloatingPointError('Nonfinite channel model or optimizer at update 25')
            if (spatial_study or fem_study) and offset in (0,24,updates-1):
                memory=jax.devices()[0].memory_stats()
                with (run/'spatial_memory.jsonl').open('a') as output:
                    output.write(json.dumps(dict(step=i+1,allocator=memory),allow_nan=False)+'\n')
            if not np.isfinite(float(total)):raise FloatingPointError(f'Nonfinite loss at update {i+1}')
            if feedback_causal:
                feedback_record=dict(step=i+1,epsilon=float(epsilon_used),
                    next_epsilon=float(causal_aux['next_epsilon']),
                    mean_weights=np.asarray(causal_aux['mean_weights']).tolist(),
                    last_time_weights=np.asarray(causal_aux['weights'][:,-1]).tolist())
                with (run/'causal-controller.jsonl').open('a') as output:
                    output.write(json.dumps(feedback_record,allow_nan=False)+'\n')
                epsilon=causal_aux['next_epsilon']
            if (i==start_step or (i+1)%100==0 or (causal and i+1==c.train.iterations)
                    or (causal and not feedback_causal and i%c.causal.steps_per_epsilon==0)):
                record=dict(step=i+1,total=float(total),parts=dict(zip(part_names(c),np.asarray(parts).tolist(),strict=True)))
                if causal:
                    weights=np.asarray(causal_aux['weights']);segments=np.asarray(causal_aux['segments'])
                    raw=np.asarray(causal_aux['unweighted_parts'])
                    record['unweighted_parts']=dict(zip(part_names(c),raw.tolist(),strict=True))
                    record['causal']=dict(epsilon=float(epsilon_used),mean_weight=float(weights.mean()),
                        minimum_weight=float(weights.min()),last_time_weights=weights[:,-1].tolist())
                    causal_record=dict(step=i+1,total=float(total),epsilon=float(epsilon_used),
                        weights=weights.tolist(),segments=segments.tolist(),unweighted_parts=raw.tolist())
                    if feedback_causal:
                        record['causal']['next_epsilon']=float(epsilon)
                        causal_record['next_epsilon']=float(epsilon)
                    with (run/'causal.jsonl').open('a') as output:
                        output.write(json.dumps(causal_record,allow_nan=False)+'\n')
                if adaptive:record['weights']=dict(zip(group_names(c),np.asarray(adaptive_weights).tolist(),strict=True))
                if staggered:record.update(stage=stage,block_active=list(active))
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(json.dumps(record),flush=True)
            if refresh:
                balance_record=dict(step=i+1,groups=group_names(c),weights=np.asarray(adaptive_weights).tolist(),
                    gradient_l1_norms=np.asarray(norms).tolist())
                with (run/'balance.jsonl').open('a') as output:
                    output.write(json.dumps(balance_record,allow_nan=False)+'\n')
            if staggered and (i%c.stagger.switch_every==0 or i+1==c.train.iterations):
                with (run/'stagger.jsonl').open('a') as output:
                    output.write(json.dumps(dict(step=i+1,stage=stage,block_active=list(active)),allow_nan=False)+'\n')
            if (i+1)%c.log.eval_every==0 or i+1==c.train.iterations:metrics=assess(i+1)
            if (i+1)%c.log.checkpoint_every==0 or (staggered and i+1==c.stagger.alternating_steps):
                eqx.tree_serialise_leaves(run/f'ckpt_{i+1:06d}.eqx',net)
        if online:
            pool_state=pool.save_state(c.train.iterations)
            np.savez_compressed(run/'known_inputs_final.npz',**{k:np.asarray(v) for k,v in bank.items()})
            online_summary=pool.summary()
            online_summary['pool_state']=str(pool_state.resolve())
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    eqx.tree_serialise_leaves(run/'optimizer_final.eqx',state)
    if adaptive:np.savez_compressed(run/'balance_final.npz',weights=np.asarray(adaptive_weights))
    sample_record=dict(indices=sampled,times=sampled_times)
    if discrete:sample_record['intervals']=sampled_intervals
    if online:sample_record.update(draw_ids=sampled_draw_ids,amplitudes=sampled_amplitudes)
    if compact:
        sample_record.update(core_ids=core_ids,spatial_pool_ids=pool_ids,halo_counts=halo_counts)
    np.savez_compressed(run/'sampling.npz',**sample_record)
    if spatial_study or fem_study:
        np.savez_compressed(run/'spatial_timing.npz',update=timings,batch_preparation=spatial_seconds)

    np.savez_compressed(run/'prediction.npz',fields=prediction,points=reference['points'],
        time=reference['time'],amplitudes=reference['amplitudes'],reference=reference['fields'])
    time_method='BE-started BDF2 time' if discrete else 'time JVP'
    summary=dict(status='completed',wall_seconds=time.perf_counter()-start,
        warm_update_seconds=float(np.mean(timings[1:])),metrics=metrics,
        iterations=c.train.iterations,train_conditions=c.operator.train_conditions,
        supervised_evolved_field_samples=0,
        method=(f'Streamfunction AD (including its spatial Hessian), {time_method}, GMLS physical Laplacians'
                if 'velocity' in c else f'Spatial AD first derivatives, {time_method}, GMLS second derivatives'),
        note='Experimental differentiation stencil; not regional pair-kernel NS or validated mesh independence')
    if continuation is not None:
        summary.update(start_step=start_step,new_updates=updates,parent_run=continuation['parent_run'],
            cumulative_wall_seconds=continuation['parent_cumulative_wall_seconds']+summary['wall_seconds'])
    if initialization is not None:summary['initialization']=initialization
    memory=jax.devices()[0].memory_stats()
    if memory is not None:summary['device_memory_bytes']=memory
    summary['device']=dict(kind=jax.devices()[0].device_kind,platform=jax.default_backend(),jax_version=jax.__version__)
    if adaptive:summary['balance']=dict(groups=group_names(c),final_weights=np.asarray(adaptive_weights).tolist())
    if adaptive and 'grouping' in c.balance:
        summary['balance'].update(settings=c.balance.to_dict(),
            algorithm='Stopped inverse-L1-gradient ratio with clipping and EMA; not original learned GradNorm',
            gradient_scope='Norm of each full condition/time-mean parameter gradient, before optimizer clipping')
    if 'microbatch' in c.train:
        summary['microbatch']=dict(**c.train.microbatch.to_dict(),
            time_samples_per_update=c.sample.Nt,optimizer_updates_per_batch=1,
            clipping='Once, after the complete averaged gradient')
    if causal:
        summary['causal']=dict(**c.causal.to_dict(),final_epsilon=float(epsilon_used),
            final_mean_weight=float(np.asarray(causal_aux['weights']).mean()),
            scope='Stopped prefixes across all sampled times and all configured soft terms per geometry; no full-time convergence guarantee')
        if feedback_causal:summary['causal']['next_epsilon']=float(epsilon)
    if spatial_study:
        warm=slice(10,None)
        summary['spatial']=dict(**c.sample.spatial.to_dict(),
            first_update_seconds=timings[0],warm_update_median_seconds=float(np.median(timings[warm])),
            warm_batch_median_seconds=float(np.median(spatial_seconds[warm])),
            warm_step_with_batch_median_seconds=float(np.median(
                np.asarray(timings[warm])+np.asarray(spatial_seconds[warm]))),
            spatial_producer_wait_seconds=spatial.wait_seconds if compact else 0.,
            total_process_seconds=time.perf_counter()-entry_start)
        if compact:summary['spatial']['actual_halo_median']=float(np.median(halo_counts))
    if 'boundary_sampling' in c:summary['boundary_sampling']=c.boundary_sampling.to_dict()
    if online:summary['online']=online_summary
    if staggered:summary['stagger']=dict(update_counts=stage_counts,
        scope='Loss/head alternation with lagged cross-physics factors; shared trunk and global SOAP state; physical fields not fully frozen')
    if 'phase_metric' in c:summary['phase_metric']=c.phase_metric.to_dict()
    if 'phase_coordinate' in c:summary['phase_coordinate']=c.phase_coordinate.to_dict()
    if 'residual_transform' in c:summary['residual_transform']=c.residual_transform.to_dict()
    if discrete:summary['time_discretization']=c.time_discretization.to_dict()
    if 'time_derivative' in c:summary['time_derivative']=c.time_derivative.to_dict()
    if 'stream_lifting' in c:summary['stream_lifting']=c.stream_lifting.to_dict()
    if 'chemical_residual_scale'in c.loss:
        summary['chemical_scaling']=dict(output_amplitude=c.loss.chemical_scale,
            residual_normalizer=c.loss.chemical_residual_scale,boundary_normalizer=c.loss.boundary_mu_scale,
            scope='Physical mu output amplitude, chemical residual normalizer and boundary normalizer are independent explicit settings')
    if 'phase_transport' in c:
        summary['phase_transport']=c.phase_transport.to_dict()
        if c.phase_transport.kind in ('gmls_conservative','integral_conservative'):
            summary['derivatives']=c.derivatives.to_dict()
            spatial='Moment-corrected radial integral'if c.nl.method=='integral_moment'else 'GMLS'
            summary['method']=(spatial+' PDE/BC first derivatives and physical Laplacians; '
                +spatial+' div(phi*v) without an exact discrete mass identity; '+time_method+
                '; unchanged AD streamfunction physical velocity/constraints; SBP loss quadrature only')
        else:
            summary['method']+='; phase advection from mapped conservative SBP curl fluxes, SBP loss quadrature'
    if 'stress_formulation' in c:
        summary['stress_formulation']=c.stress_formulation.to_dict()
        summary['method']='Mixed Cauchy stress with spatial AD stress gradients; streamfunction AD velocity/Hessian; '+time_method+'; GMLS phi/mu Laplacians'
        if 'phase_transport' in c:summary['method']+='; conservative SBP phase advection'
    if 'chemical_constraint'in c:
        summary['chemical_constraint']=c.chemical_constraint.to_dict()
        summary['method']+='; exact interior chemical recovery, retained decoder boundary traces, no chemical soft loss'
    if fem_study:
        summary['method'] = ('Isoparametric P2/P1 finite-element weak NS--CH; natural phase/mu flux and outlet traction; '
                             'conservative outlet phase transport; unchanged AD streamfunction and time JVP')
        summary['note'] = 'Approximate mass-diagonal dual residual norm; not a direct numerical FEM solve'
        summary['fem'] = dict(**c.nl.to_dict(), full_coordinate_nodes=(c.sample.Nx+1)*(c.sample.Ny+1),
            warm_update_median_seconds=float(np.median(timings[10:])),
            warm_step_with_batch_median_seconds=float(np.median(np.asarray(timings[10:])+np.asarray(spatial_seconds[10:]))),
            first_update_seconds=timings[0], total_process_seconds=time.perf_counter()-entry_start,
            boundary_penalties=False, continuity='Unchanged analytic streamfunction; FE interpolant only approximately divergence-free')
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main(_):train(with_run_seed(_CONFIG.value,_SEED.value))


if __name__=='__main__':app.run(main)
