"""Physics-only random-ceramic-geometry PINO; common local-reference evaluation."""
import argparse
from contextlib import ExitStack
from functools import partial
import importlib.util
import json
from pathlib import Path
import secrets
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy
from .evaluate import load_references,make_evaluator,render_gallery,render_convergence
from .network import build_network
from .problem import validate


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_bank(path,c):
    path=Path(path).resolve();manifest=json.loads((path/'manifest.json').read_text())
    actual=json.loads(c.to_json())
    for name in ('geometry','particles','physical','mesh'):
        if manifest['config'][name]!=actual[name]:raise ValueError(f'Geometry bank {name} differs')
    for name in ('delta','visibility_points','visibility_tolerance'):
        if manifest['config']['nl'][name]!=actual['nl'][name]:raise ValueError(f'Bank nonlocal {name} differs')
    for name in ('K_pool','K_core'):
        if manifest['config']['sample'][name]!=actual['sample'][name]:raise ValueError(f'Bank {name} differs')
    for name in ('domain_shape','wall_points_per_particle','wall_points_per_side','mouth_points','ic_points','geometry_count'):
        if manifest['config']['operator'][name]!=actual['operator'][name]:raise ValueError(f'Bank conditioning {name} differs')
    with np.load(path/'inputs.npz',allow_pickle=False) as source:
        arrays={key:source[key] for key in source.files}
    if set(arrays)!=set(manifest['fixed_keys']+manifest['graph_keys']):raise ValueError('Bank payload schema mismatch')
    n=len(manifest['geometries']);pool=c.sample.K_pool
    for name,value in arrays.items():
        if value.shape[0]!=n or not np.isfinite(value).all():raise ValueError(f'Invalid bank array {name}')
        if name in manifest['graph_keys'] and value.shape[1]!=pool:raise ValueError(f'Invalid pool axis {name}')
    for name in ('domain_weights','boundary_weights','wall_weights','mouth_weights','ic_weights','pde_weights','coefficients'):
        if np.any(arrays[name]<0):raise ValueError(f'Negative physical weight: {name}')
        if name!='coefficients' and np.any(arrays[name].sum(axis=-1)<=0):
            raise ValueError(f'Empty physical measure: {name}')
    for name in ('neighbors','core_indices'):
        if not np.issubdtype(arrays[name].dtype,np.integer):raise ValueError(f'Integer indices required: {name}')
    if np.max(arrays['neighbors'])>=arrays['graph_points'].shape[2] or np.min(arrays['neighbors'])<0:
        raise ValueError('Out-of-range halo index')
    if np.max(arrays['core_indices'])>=arrays['graph_points'].shape[2] or np.min(arrays['core_indices'])<0:
        raise ValueError('Out-of-range core index')
    if np.any(np.sum(arrays['domain_weights'],axis=-1)<=0):raise ValueError('Empty domain measure')
    return arrays,manifest


def train(c,output,inputs,reference,config_path,paired_run,
          sampling_control=None,exclude_references=(),full_grid_control=None):
    started=time.perf_counter();validate(c)
    from .full_grid import is_full,prepare_initial_full_bank,validate_full
    full_grid=is_full(c)
    if full_grid:validate_full(c)
    if sum(path is not None for path in (paired_run,sampling_control,full_grid_control))>1:
        raise ValueError('Select exactly one comparison control, not multiple controls')
    if full_grid_control is not None and not full_grid:
        raise ValueError('--full-grid-control requires the explicit sample.full_grid configuration')
    if c.train.compute_dtype!='float32':raise ValueError('This initial runner qualifies ordinary FP32 only')
    if c.train.iterations<1 or c.operator.condition_batch<1:raise ValueError('Positive update/batch counts required')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    bank,manifest=load_bank(inputs,c)
    references,reference_manifest=load_references(reference,c)
    online_enabled='online' in c.operator
    forbidden=[entry['geometry'] for entry in reference_manifest['cases']]
    if online_enabled:
        from .online import RollingPool,pad_initial_bank,validate_online
        validate_online(c)
        if paired_run is not None:
            raise ValueError('Online sampling control uses --sampling-control, not the spatial-method --paired-run')
        for path in exclude_references:
            excluded=json.loads((Path(path).resolve(strict=True)/'manifest.json').read_text())
            for name in ('geometry','particles'):
                if excluded['config'][name]!=json.loads(c.to_json())[name]:
                    raise ValueError(f'Excluded reference has a different geometry prior: {path}/{name}')
            forbidden.extend(entry['geometry'] for entry in excluded['cases'])
        if full_grid:
            bank=prepare_initial_full_bank(bank,c,manifest,c.operator.online.workers)
        else:
            bank=pad_initial_bank(bank,c)
    elif sampling_control is not None or exclude_references or full_grid:
        raise ValueError('Sampling control/excluded geometry arguments require operator.online')
    if c.nl.method=='local_ad':
        from .local_ad import make_loss_vector,LOSS_NAMES
    else:
        from .losses import make_loss_vector,LOSS_NAMES
    from .losses import make_step
    control=None
    if paired_run is not None:
        paired_run=Path(paired_run).resolve()
        prior=json.loads((paired_run/'config.json').read_text())
        current=json.loads(c.to_json())
        prior_method=prior['nl'].pop('method');current_method=current['nl'].pop('method')
        prior_bias=prior['loss'].pop('gradnorm_bias');current_bias=current['loss'].pop('gradnorm_bias')
        if prior!=current or prior_bias[:5]!=current_bias[:5] or {prior_method,current_method}!={'nonlocal','local_ad'}:
            raise ValueError('Paired control permits only method and explicit extra Neumann biases')
        runtime=json.loads((paired_run/'runtime.json').read_text())
        if runtime['inputs']!=str(Path(inputs).resolve()) or runtime['reference']!=str(Path(reference).resolve()):
            raise ValueError('Paired run must use identical bank and common reference')
        seed=runtime['seed']
        with np.load(paired_run/'sampling.npz',allow_pickle=False) as source:
            control={key:source[key] for key in source.files}
    elif sampling_control is not None:
        from .online_validation import validate_fixed_control
        sampling_control=Path(sampling_control).resolve(strict=True)
        runtime=validate_fixed_control(c,sampling_control,inputs,reference)
        seed=runtime['seed']
        with np.load(sampling_control/'sampling.npz',allow_pickle=False) as source:
            control={key:source[key] for key in source.files}
    elif full_grid_control is not None:
        from .full_grid_validation import validate_full_control
        full_grid_control=Path(full_grid_control).resolve(strict=True)
        runtime=validate_full_control(c,full_grid_control,inputs,reference)
        seed=runtime['seed']
        with np.load(full_grid_control/'sampling.npz',allow_pickle=False) as source:
            control={key:source[key] for key in source.files}
    else:seed=secrets.randbits(32)
    rng=np.random.default_rng(seed)
    condition_ids=rng.integers(len(manifest['geometries']),size=(c.train.iterations,c.operator.condition_batch),dtype=np.int32)
    pool_ids=rng.integers(c.sample.K_pool,size=condition_ids.shape,dtype=np.int32)
    times=rng.uniform(0,c.sample.t_max,size=(c.train.iterations,c.sample.Nt)).astype(np.float32)
    trace=dict(condition_ids=condition_ids,pool_ids=pool_ids,times=times)
    if control is not None:
        for key in trace:
            if not np.array_equal(trace[key],control[key]):raise ValueError(f'Paired draw mismatch: {key}')
    net=build_network(c,jax.random.PRNGKey(seed))
    optimizer=make_optimizer(c);state=optimizer.init(eqx.filter(net,eqx.is_array))
    initialization_control=next((path for path in (paired_run,sampling_control,full_grid_control)
                                 if path is not None),None)
    if initialization_control is not None:
        old_net=eqx.tree_deserialise_leaves(initialization_control/'model_initial.eqx',net)
        old_state=eqx.tree_deserialise_leaves(initialization_control/'optimizer_initial.eqx',state)
        for new,old in zip(jax.tree.leaves((net,state)),jax.tree.leaves((old_net,old_state)),strict=True):
            if eqx.is_array(new) and not np.array_equal(np.asarray(new),np.asarray(old)):
                raise ValueError('Cold paired model/optimizer do not reproduce saved initialization')
    loss=make_loss_vector(c);step=make_step(loss,optimizer,c,LOSS_NAMES)
    weights=jnp.ones(len(LOSS_NAMES),dtype=jnp.float32)
    run=Path(output).resolve();run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',json.loads(c.to_json()))
    write_json(run/'runtime.json',dict(seed=seed,inputs=str(Path(inputs).resolve()),
        reference=str(Path(reference).resolve()),devices=[str(x) for x in jax.devices()],
        cold_initialization=True,paired_run=None if paired_run is None else str(paired_run),
        sampling_control=None if sampling_control is None else str(sampling_control),
        full_grid_control=None if full_grid_control is None else str(full_grid_control),
        pde_sampling='full_grid' if full_grid else 'core_halo',
        pool_ids_used_for_pde=not full_grid,
        rolling_inputs=online_enabled,
        excluded_reference_paths=[str(Path(path).resolve()) for path in exclude_references],
        paired_sampling_verified=control is not None,training_uses_reference_targets=False,
        checkpoint_selection='fixed training budget',conditioning='known domain and wall point measures',
        parameters=sum(x.size for x in jax.tree.leaves(net) if eqx.is_array(x))))
    np.savez_compressed(run/'sampling.npz',**trace)
    if online_enabled:
        np.savez_compressed(run/'online_initial_inputs.npz',**bank)
        write_json(run/'excluded_geometries.json',forbidden)
    eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    eqx.tree_serialise_leaves(run/'optimizer_initial.eqx',state)
    repo=Path(__file__).resolve().parents[2]
    sources=list(Path(__file__).parent.rglob('*.py'))+[
        repo/'src/nonlocal_pinn/nn/point_conditional.py',repo/'src/nonlocal_pinn/nn/conditional.py',
        repo/'src/nonlocal_pinn/training/optimizers.py',repo/'src/nonlocal_pinn/training/runtime.py']
    for source in sources:
        target=run/'source'/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(config_path,run/'supplied_config.py')
    # Keep the bank on device; indexed batches preserve fixed traced shapes.
    gpu_bank={key:jnp.asarray(value) for key,value in bank.items()}
    @partial(jax.jit,donate_argnums=(0,))
    def replace_slots(current,slots,replacements):
        return {key:value.at[slots].set(replacements[key]) for key,value in current.items()}
    fixed_keys=manifest['fixed_keys'];graph_keys=manifest['graph_keys']
    @jax.jit
    def gather(bank_arrays,ids,pools):
        data={key:bank_arrays[key][ids] for key in fixed_keys}
        selected=jnp.zeros_like(pools) if full_grid else pools
        data.update({key:bank_arrays[key][ids,selected] for key in graph_keys})
        if c.nl.method=='local_ad':
            # Local residuals evaluate only the same cores, never the unused halo.
            data['graph_points']=jax.vmap(lambda points,core:points[core])(data['graph_points'],data['core_indices'])
            data['core_indices']=jnp.broadcast_to(jnp.arange(c.sample.K_core,dtype=jnp.int32),ids.shape+(c.sample.K_core,))
            del data['neighbors'];del data['coefficients']
        return data
    evaluator=make_evaluator(c,references)
    history=[];updates=[];sample_seconds=[];eval_seconds=[];memory=[];train_start=time.perf_counter()
    online_seconds=[];online_summary=None
    if online_enabled:
        geometry_draws=np.empty_like(condition_ids,dtype=np.int64)
        spatial_versions=np.empty_like(condition_ids,dtype=np.int64)
    def evaluate(iteration,model):
        begin=time.perf_counter();predictions,entry=evaluator(model)
        entry.update(step=iteration,seconds=time.perf_counter()-train_start);history.append(entry)
        with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in entry.items() if k!='conditions'},allow_nan=False),flush=True)
        eval_seconds.append(time.perf_counter()-begin)
        render_convergence(run/'convergence.png',history)
        return predictions,entry
    with ExitStack() as stack:
        pool=stack.enter_context(RollingPool(c,run,manifest,seed,forbidden)) if online_enabled else None
        predictions,last=evaluate(0,net)
        stream=stack.enter_context((run/'loss.jsonl').open('w'))
        for i in range(c.train.iterations):
            if pool is not None:
                refresh_start=time.perf_counter()
                replacement=pool.before_step(i)
                if replacement is not None:
                    slots,values=replacement
                    gpu_bank=replace_slots(gpu_bank,jnp.asarray(slots),
                        {key:jnp.asarray(value) for key,value in values.items()})
                    jax.block_until_ready(gpu_bank)
                online_seconds.append(time.perf_counter()-refresh_start)
                geometry_draws[i]=pool.draw_ids[condition_ids[i]]
                spatial_versions[i]=pool.spatial_versions[condition_ids[i]]
            sample_start=time.perf_counter()
            payload=gather(gpu_bank,jnp.asarray(condition_ids[i]),jnp.asarray(pool_ids[i]))
            jax.block_until_ready(payload)
            sample_seconds.append(time.perf_counter()-sample_start)
            begin=time.perf_counter()
            net,state,weights,values,finite=step(net,state,weights,jnp.asarray(times[i]),payload,
                jnp.asarray(i%c.train.gradnorm_update_every==0))
            jax.block_until_ready((net,state,weights,values,finite));updates.append(time.perf_counter()-begin)
            if not bool(finite):raise FloatingPointError(f'Nonfinite objective/gradient at update{i+1}')
            if i in (0,24,c.train.iterations-1):
                stats=jax.devices()[0].memory_stats()
                memory.append(dict(step=i+1,allocator=None if stats is None else {k:int(v) for k,v in stats.items()}))
                write_json(run/'memory.json',memory)
            if i==0 or (i+1)%c.log.log_every==0 or i+1==c.train.iterations:
                entry=dict(step=i+1,parts=np.asarray(values).tolist(),loss_names=list(LOSS_NAMES),
                    weights=np.asarray(weights).tolist(),update_seconds=updates[-1],
                    elapsed_seconds=time.perf_counter()-train_start)
                stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush();print(json.dumps(entry),flush=True)
            if (i+1)%c.log.checkpoint_every==0:
                eqx.tree_serialise_leaves(run/f'model_{i+1:06d}.eqx',net)
                if pool is not None:pool.save_state(i+1)
            if (i+1)%c.log.eval_every==0 or i+1==c.train.iterations:
                predictions,last=evaluate(i+1,net)
        if pool is not None:
            if c.train.iterations%c.log.checkpoint_every:pool.save_state(c.train.iterations)
            np.savez_compressed(run/'online_sampling.npz',geometry_draw_ids=geometry_draws,
                                spatial_versions=spatial_versions)
            np.savez_compressed(run/'online_final_inputs.npz',
                                **{key:np.asarray(value) for key,value in gpu_bank.items()})
    if online_enabled:
        online_summary=pool.summary()
        write_json(run/'online_summary.json',online_summary)
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    eqx.tree_serialise_leaves(run/'optimizer_final.eqx',state)
    for i,(pred,ref) in enumerate(zip(predictions,references,strict=True)):
        np.savez_compressed(run/f'prediction_{i:03d}.npz',prediction=pred,points=ref['points'],times=ref['times'])
    render_gallery(run/'fields.png',references,predictions,f'{c.nl.method}: unseen ceramic geometries')
    write_json(run/'timing.json',dict(first_update_compile_seconds=updates[0],
        warm_update_mean_seconds=None if len(updates)<2 else float(np.mean(updates[1:])),
        update_seconds=updates,sample_seconds=sample_seconds,evaluation_seconds=eval_seconds,setup_seconds=train_start-started,
        online_update_seconds=online_seconds,
        total_seconds=time.perf_counter()-started))
    summary=dict(steps=c.train.iterations,final=last,loss_names=list(LOSS_NAMES),
        geometry_count=len(manifest['geometries']),heldout_count=len(references),
        spatial_derivative=c.nl.method,training_uses_reference_targets=False,
        online=online_summary,
        final_weights=np.asarray(weights).tolist(),total_seconds=time.perf_counter()-started)
    write_json(run/'summary.json',summary);print('COMPLETE '+str(run),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/train_default.py')
    parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--paired-run',type=Path)
    parser.add_argument('--sampling-control',type=Path,
                        help='Completed fixed-input run with identical recipe; reproduce cold state and common random draws')
    parser.add_argument('--exclude-reference',action='append',type=Path,default=[],
                        help='Additional reference directory whose geometry must never enter the rolling training pool')
    parser.add_argument('--full-grid-control',type=Path,
                        help='Completed rolling core/halo run; match cold state and all geometry/IC/time draws')
    args=parser.parse_args();path=args.config.resolve(strict=True)
    spec=importlib.util.spec_from_file_location('particle_training_config',path)
    if spec is None or spec.loader is None:raise ValueError(f'Invalid config{path}')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train(module.get_config(),args.output,args.inputs,args.reference,path,args.paired_run,
          args.sampling_control,args.exclude_reference,args.full_grid_control)


if __name__=='__main__':main()
