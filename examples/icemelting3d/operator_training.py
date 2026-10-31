"""Online random-IC volumetric neural operator through melting train.py."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.icemelting3d.ellipsoids import image_points,signed_distance,validate_parameters
from examples.icemelting3d.operator_data import build_network,load_dataset,make_sampler,physics,validate_config
from examples.icemelting3d.operator_losses import make_loss_fn


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def build_graph(c):
    n=c.sample.Nx
    axis=np.linspace(0,1,n,dtype=np.float32)
    points=np.stack(np.meshgrid(axis,axis,axis,indexing='xy'),-1).reshape(-1,3)
    x=jnp.asarray(points)
    op=build_D(x,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
    expected_edges=6*(n-1)*n*n
    if len(op._vals)!=expected_edges or op._k_max!=6:
        raise ValueError('Actual melting graph does not have the declared six-axis neighborhood')
    from examples.icemelting3d.operator_reference import coefficients
    weight=coefficients(c)[3]
    if not np.allclose(op._vals_np,-weight,rtol=2e-6,atol=0):
        raise ValueError('Actual cubic weights disagree with independent six-neighbor formula')
    info=dict(grid_nodes=len(x),directed_edges=len(op._vals),degree_max=op._k_max,
        independent_weight=weight,actual_weight_min=float(-op._vals_np.max()),
        actual_weight_max=float(-op._vals_np.min()))
    return op,x,info


def build_pool(c):
    op,x,info=build_graph(c)
    pool=[(x[ids],sub,mask) for ids,sub,mask,_ in op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)]
    info.update(padded_queries=len(pool[0][0]),padded_edges=len(pool[0][1][0]),
        core_points=c.sample.K_core,pool_entries=len(pool))
    return pool,info


def make_step(c,optimizer):
    loss=make_loss_fn(c)
    @eqx.filter_jit
    def step(net,state,times,images,u0,points,sub,mask):
        value,grad=eqx.filter_value_and_grad(loss)(net,times,images,u0,points,sub,mask)
        finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)]))
        updates,state=optimizer.update(grad,state,net)
        return eqx.apply_updates(net,updates),state,value,finite
    return step


def make_predictor(c):
    width=physics(c)[3]
    encode=eqx.filter_jit(lambda n,image:n.encode(image))
    @eqx.filter_jit
    def decode(n,encoded,points,times,u0):
        prepared=n.prepare_encoded(encoded,points,jnp.zeros((len(points),0),dtype=points.dtype))
        return jnp.tanh(u0[None,:]+times[:,None]*n.decode(prepared,times)[...,0])
    initial_fn=eqx.filter_jit(lambda points,p:signed_distance(points,p,c.initial.sdf_iterations)/width)
    image_fn=eqx.filter_jit(lambda points,p:jnp.tanh(signed_distance(points,p,c.initial.sdf_iterations)/width))
    input_points=jnp.asarray(image_points(tuple(c.arch.image_shape)),dtype=jnp.float32)
    def predict(net,parameters,points,times):
        p=jnp.asarray(parameters,dtype=jnp.float32)
        image=image_fn(input_points,p).reshape(tuple(c.arch.image_shape))
        encoded=encode(net,image)
        result=[]
        chunk=c.operator.evaluation_chunk
        for start in range(0,len(points),chunk):
            actual=np.asarray(points[start:start+chunk],dtype=np.float32)
            padded=np.pad(actual,((0,chunk-len(actual)),(0,0)),mode='edge')
            x=jnp.asarray(padded);u0=initial_fn(x,p)
            output=np.asarray(decode(net,encoded,x,jnp.asarray(times,dtype=jnp.float32),u0))
            result.append(output[:,:len(actual)])
        return np.concatenate(result,axis=1)
    return predict


def field_metrics(predicted,target):
    if not np.isfinite(predicted).all():
        raise FloatingPointError('Nonfinite physical operator evaluation')
    solid=(target+1)/2
    error=predicted-target
    def ratio(a,b):
        norm=float(np.linalg.norm(b))
        return None if norm==0 else float(np.linalg.norm(a)/norm)
    band=abs(target)<.9
    return dict(pooled_phi=ratio(error,target),terminal_phi=ratio(error[-1],target[-1]),
        pooled_solid=ratio(error/2,solid),terminal_solid=ratio(error[-1]/2,solid[-1]),
        interface_phi_absolute_rmse=None if not band.any() else float(np.sqrt(np.mean(error[band]**2))),
        phi_by_time=[ratio(e,r) for e,r in zip(error,target)],
        solid_by_time=[ratio(e/2,s) for e,s in zip(error,solid)],
        solid_fraction_predicted=((predicted+1)/2).mean(axis=1).tolist(),
        solid_fraction_target=solid.mean(axis=1).tolist(),
        phi_min=float(predicted.min()),phi_max=float(predicted.max()),
        zero_target_norm_policy='Relative metric is null if reference norm is exactly zero; no denominator floor')


def train_operator(c):
    validate_config(c)
    if 'spatial' in c.sample:
        from examples.icemelting3d.spatial_training import train_spatial
        return train_spatial(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if c.operator.action=='prepare':
        from examples.icemelting3d.operator_reference import prepare_dataset
        return prepare_dataset(c)
    jax.config.update('jax_enable_x64',False)
    if jax.default_backend()!='gpu':
        raise ValueError('Train/memory actions require the declared GPU backend')
    start=time.perf_counter()
    run_name=c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-ellipsoid-'+c.operator.action
    run=Path(c.log.log_dir)/run_name
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',json.loads(c.to_json()))
    repo=Path(__file__).resolve().parents[2]
    sources=[p for p in (repo/'examples/icemelting3d').glob('*.py')]
    sources+=list((repo/'examples/icemelting3d/configs').glob('*.py'))
    sources+=[repo/p for p in ('src/nonlocal_pinn/nn/conditional.py','src/nonlocal_pinn/training/optimizers.py',
        'src/nonlocal_pinn/training/runtime.py','src/nonlocal_pinn/operators/__init__.py')]
    names=[]
    for source in sources:
        name=source.relative_to(repo);target=run/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);names.append(str(name))
    write_json(run/'source/files.json',names)
    pool,graph_info=build_pool(c);write_json(run/'graph.json',graph_info)
    sampler=make_sampler(c)
    keys=jax.random.split(jax.random.key(c.train.seed),3)
    net=build_network(c,keys[0]);eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    optimizer=make_optimizer(c);state=optimizer.init(eqx.filter(net,eqx.is_array))
    step=make_step(c,optimizer)
    setup=time.perf_counter()-start
    records=[];update_times=[];sample_times=[]
    if c.operator.action=='memory':
        iterations=c.operator.probe_steps
        refs=None;manifest=None;predict=None
    else:
        iterations=c.train.iterations
        manifest,refs=load_dataset(c);predict=make_predictor(c)
    parameters_record=np.empty((iterations,c.operator.condition_batch,10),dtype=np.float32)
    times_record=np.empty((iterations,c.sample.Nt),dtype=np.float32)
    writer=SummaryWriter(str(run))
    evaluation_history=[]
    def evaluate(indices,iteration,split):
        begin=time.perf_counter();items=[]
        for index in indices:
            ref=refs[index]
            predicted=predict(net,ref['parameters'],ref['points'],ref['time'])
            item=dict(index=index,**field_metrics(predicted,ref['phi']));items.append(item)
            if iteration==iterations:
                slice_prediction=predict(net,ref['parameters'],ref['slice_points'],ref['time'])
                np.savez_compressed(run/f'prediction_{index:03d}.npz',prediction=predicted,target=ref['phi'],
                    parameters=ref['parameters'],points=ref['points'],time=ref['time'],
                    slice_prediction=slice_prediction,slice_target=ref['slice_phi'],slice_points=ref['slice_points'])
        entry=dict(step=iteration,split=split,conditions=items,seconds=time.perf_counter()-begin,
            pooled_phi=float(np.mean([item['pooled_phi'] for item in items])),
            terminal_phi=float(np.mean([item['terminal_phi'] for item in items])))
        with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        writer.add_scalar('eval/'+split+'_pooled_phi',entry['pooled_phi'],iteration)
        writer.add_scalar('eval/'+split+'_terminal_phi',entry['terminal_phi'],iteration)
        evaluation_history.append(entry)
        print('EVALUATION',json.dumps({k:v for k,v in entry.items() if k!='conditions'}),flush=True)
        return entry
    if refs is not None:evaluate(manifest['monitor_indices'],0,'monitor')
    loop_start=time.perf_counter()
    for k in range(iterations):
        points,sub,mask=pool[k%len(pool)]
        times=jax.random.uniform(jax.random.fold_in(keys[1],k),(c.sample.Nt,),minval=c.sample.t_min,maxval=1.)
        sample_start=time.perf_counter()
        images,u0,parameters=sampler(jax.random.fold_in(keys[2],k),points)
        jax.block_until_ready((images,u0,parameters))
        sample_times.append(time.perf_counter()-sample_start)
        parameters_record[k]=np.asarray(parameters);times_record[k]=np.asarray(times)
        validate_parameters(parameters_record[k])
        update_start=time.perf_counter()
        net,state,value,finite=step(net,state,times,images,u0,points,sub,mask)
        jax.block_until_ready((net,state,value,finite))
        update_times.append(time.perf_counter()-update_start)
        if not bool(finite):raise FloatingPointError(f'Nonfinite latent residual/gradient at step{k+1}')
        if k==0 or (k+1)%c.log.log_every==0 or k+1==iterations:
            record=dict(step=k+1,loss=float(value),seconds=time.perf_counter()-loop_start,
                allocator={name:int(v) for name,v in jax.devices()[0].memory_stats().items()})
            records.append(record);print('TRAIN',json.dumps(record),flush=True)
            writer.add_scalar('loss/pde',record['loss'],k+1)
        if refs is not None and ((k+1)%c.log.eval_every==0 or k+1==iterations):
            evaluate(manifest['monitor_indices'],k+1,'monitor')
    elapsed=time.perf_counter()-loop_start
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    np.savez_compressed(run/'sampling.npz',parameters=parameters_record,times=times_record)
    write_json(run/'loss.json',records)
    test=None if refs is None else evaluate(manifest['test_indices'],iterations,'test')
    stats={name:int(v) for name,v in jax.devices()[0].memory_stats().items()}
    summary=dict(run=str(run),action=c.operator.action,iterations=iterations,
        condition_batch=c.operator.condition_batch,setup_seconds=setup,trainer_seconds=elapsed,
        first_update_compile_and_call_seconds=update_times[0],
        warm_update_mean_seconds=None if iterations<2 else float(np.mean(update_times[1:])),
        warm_sampling_mean_seconds=None if iterations<2 else float(np.mean(sample_times[1:])),
        allocator=stats,graph=graph_info,test=test,positive_time_training_labels=False,
        scope='Online random ellipsoids; input volume encoder, coordinate-only queries; hard IC/no motion prior; native graph reference, not continuum validation')
    write_json(run/'summary.json',summary)
    write_json(run/'timing.json',dict(update_seconds=update_times,sampling_seconds=sample_times,evaluation=evaluation_history))
    writer.close();print('COMPLETE',json.dumps(summary),flush=True)
    return summary
