"""Physics-only finite geometry-bank CH operator with fixed padded shapes."""
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter
from nonlocal_pinn.nn.point_conditional import PointFieldOperator,HeterogeneousPointOperator
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.confined_ch2d.operator_problem import (
    validate, specification, valid_holes, case_config, condition, fixed_initial,evaluation_indices,
    sample_geometries)
from examples.confined_ch2d.problem import relative
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.sampling import temporal_samples
from examples.confined_ch2d.operator_continuation import initial_state,continuation_spec,save_checkpoint
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled,initial_operator_fields
from examples.confined_ch2d.operator_precision import compute_dtype,compute_arrays,make_compute_optimizer


def build_network(c, key):
    architecture=c.arch.to_dict()
    if 'model' in architecture:
        if architecture.pop('model')!='heterogeneous':
            raise ValueError('Unknown geometry operator architecture')
        net=HeterogeneousPointOperator(**architecture,key=key)
    else:
        net=PointFieldOperator(**architecture,key=key)
    return compute_arrays(net,c)


def load_bank(c):
    root=Path(c.operator.dataset_path)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='passed' or manifest['specification']!=specification(c):
        raise ValueError('Geometry bank must have matching explicit inputs and passed refinement')
    expected=c.operator.train_conditions+c.operator.test_conditions
    if len(manifest['conditions'])!=expected or manifest['train_indices']!=list(range(c.operator.train_conditions)) or manifest['test_indices']!=list(range(c.operator.train_conditions,expected)):
        raise ValueError('Geometry partition/count mismatch')
    monitor,labeled=evaluation_indices(manifest)
    if 'precompute' in c.operator:
        if monitor!=list(c.operator.precompute.monitor_train_indices):
            raise ValueError('Configured training monitors must match bank provenance')
        np.testing.assert_array_equal(np.asarray([r['holes'] for r in manifest['conditions']]),sample_geometries(c))
    refs=[]
    for index, record in enumerate(manifest['conditions']):
        if record['index']!=index:
            raise ValueError('Ordered unique geometry identities required')
        path=Path(record['file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Reference paths must stay inside the explicit bank')
        with np.load(root/path,allow_pickle=False) as f:
            data={key:f[key] for key in f.files}
        if 'domain_fields' in data:
            raise ValueError('domain_fields must be derived from known inputs during loading, not stored in the physical bank')
        x,v=data['points'],data['volumes'];n=len(x)
        if x.shape!=(n,2) or v.shape!=(n,) or np.any(v<=0) or not all(np.isfinite(a).all() for a in data.values()):
            raise ValueError(f'Invalid physical quadrature/arrays in geometry {index}')
        if not valid_holes(data['holes'],c) or not np.array_equal(data['holes'],np.asarray(record['holes'])):
            raise ValueError('Geometry identity/containment mismatch')
        initial=fixed_initial(x,c,xp=np)
        if index in labeled:
            if data['phi'].shape!=(c.reference.snapshots,n) or data['mu'].shape!=data['phi'].shape:
                raise ValueError('Reference field shapes mismatch')
            np.testing.assert_allclose(data['time'],np.linspace(0,c.physical.terminal_time,c.reference.snapshots),rtol=0,atol=1e-14)
            np.testing.assert_allclose(data['phi'][0],initial,rtol=0,atol=1e-14)
        else:
            if any(key in data for key in ('phi','mu','time','mass','energy')):
                raise ValueError('Operator-only training inputs must not contain evolved fields')
            np.testing.assert_array_equal(data['initial'],initial)
        data['initial']=initial
        if data['metric'].shape!=(n,n) or data['neighbors'].ndim!=2 or data['neighbors'].shape[0]!=n or data['coefficients'].shape!=data['neighbors'].shape or np.any(data['neighbors']<0) or np.any(data['neighbors']>=n) or np.any(data['coefficients']<0):
            raise ValueError(f'Operator/metric shape or range mismatch: {index}')
        features,measure=condition(data['holes'],c)
        np.testing.assert_array_equal(data['condition'],features)
        np.testing.assert_array_equal(data['condition_weights'],measure)
        cc=case_config(c,data['holes'],index)
        production=RegionalLaplacian(x,v,cc)
        neighbors=np.asarray(production.neighbors);coefficients=np.asarray(production.coefficients)
        for field in (np.ones(n),np.sin(np.arange(n)),x[:,0]+x[:,1]**2):
            a=np.sum(coefficients*(field[:,None]-field[neighbors]),axis=1)
            b=np.sum(data['coefficients']*(field[:,None]-field[data['neighbors']]),axis=1)
            np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-10)
        data['neighbors'],data['coefficients']=neighbors,coefficients
        if initial_rhs_enabled(c):data['domain_fields']=initial_operator_fields(data,c)
        refs.append(data)
    if c.operator.metric=='linear_stable_resolvent':
        from examples.confined_ch2d.operator_metrics import metric_job
        p=c.physical
        arguments=[(d['volumes'],d['neighbors'],d['coefficients'],d['initial'],
            p.beta,p.epsilon,p.mobility,p.terminal_time) for d in refs]
        reports=[]
        with ProcessPoolExecutor(max_workers=c.operator.metric_workers,
                mp_context=multiprocessing.get_context('spawn')) as executor:
            for d,(metric,report) in zip(refs,executor.map(metric_job,arguments),strict=True):
                d['metric']=metric;reports.append(report)
        manifest=dict(manifest,training_metric=dict(kind=c.operator.metric,time_scale=p.terminal_time,
            workers=c.operator.metric_workers,conditions=reports,
            scope='Frozen homogeneous-mean linearization; input-only SPD loss norm, not a PDE change'))
    nmax=max(len(d['points']) for d in refs);kmax=max(d['neighbors'].shape[1] for d in refs)
    if 'online' in c.operator:
        limits=c.operator.online
        if nmax>limits.max_points or kmax>limits.max_neighbors:
            raise ValueError(f'Initial bank exceeds online padding: {nmax}/{kmax}')
        nmax,kmax=limits.max_points,limits.max_neighbors
    bank={key:[] for key in ('points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights')}
    if initial_rhs_enabled(c):bank['domain_fields']=[]
    for d in refs:
        n=len(d['points']);k=d['neighbors'].shape[1]
        bank['points'].append(np.pad(d['points'],((0,nmax-n),(0,0))))
        bank['volumes'].append(np.pad(d['volumes'],(0,nmax-n)))
        bank['initial'].append(np.pad(d['initial'],(0,nmax-n)))
        if initial_rhs_enabled(c):bank['domain_fields'].append(np.pad(d['domain_fields'],((0,nmax-n),(0,0))))
        nb=np.repeat(np.arange(nmax)[:,None],kmax,axis=1)
        co=np.zeros((nmax,kmax));nb[:n,:k]=d['neighbors'];co[:n,:k]=d['coefficients']
        bank['neighbors'].append(nb);bank['coefficients'].append(co)
        bank['metric'].append(np.pad(d['metric'],((0,nmax-n),(0,nmax-n))))
        for key in ('condition','condition_weights'):
            bank[key].append(d[key])
    assembled={key:np.stack(value) for key,value in bank.items()}
    return manifest,refs,{key:jnp.asarray(value,dtype=compute_dtype(c) if np.issubdtype(value.dtype,np.floating) else value.dtype)
                          for key,value in assembled.items()}


def apply_operator(values, data):
    return jnp.sum(data['coefficients']*(values[:,None]-values[data['neighbors']]),axis=-1)


def encode(net,data):
    if isinstance(net,HeterogeneousPointOperator):
        domain=data['domain_fields'] if 'domain_fields' in data else data['initial'][:,None]
        inputs=(data['condition'],jnp.concatenate((data['points'],domain),axis=-1))
        return net.encode(inputs,(data['condition_weights'],data['volumes']))
    return net.encode(data['condition'],data['condition_weights'])


def prepare(net,data):
    return net.prepare_encoded(encode(net,data),data['points'])


def time_coordinate(t,c):
    """Shared decoder/lifting coordinate; the PDE JVP remains in physical time."""
    if 'phase' in c:
        tau=c.phase.initial_time_scale
        return jnp.log1p(t/tau)/jnp.log1p(c.physical.terminal_time/tau)
    return t/c.physical.terminal_time


def phase(net,prepared,t,data,c):
    t=jnp.asarray(t,dtype=compute_dtype(c))
    s=time_coordinate(t,c)
    change=net.decode(prepared,jnp.reshape(s,(1,)))[0]
    volumes=data['volumes']
    mean=jnp.dot(volumes,change)/jnp.sum(volumes)
    return jnp.where(volumes>0,data['initial']+s*(change-mean),0.)


def fields(net,data,times,c):
    data=compute_arrays(data,c)
    times=jnp.asarray(times,dtype=compute_dtype(c))
    prepared=prepare(net,data)
    def at(t):
        phi=phase(net,prepared,t,data,c)
        mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,data)
        return jnp.stack((phi,mu),axis=-1)
    return jax.vmap(at)(times)


def query_phase(net,data,points,times,c):
    """Off-grid phase queries retain the ORIGINAL physical mass quadrature."""
    data=compute_arrays(data,c)
    points=jnp.asarray(points,dtype=compute_dtype(c))
    times=jnp.asarray(times,dtype=compute_dtype(c))
    encoded=encode(net,data)
    grid=net.prepare_encoded(encoded,data['points'])
    query=net.prepare_encoded(encoded,points)
    s=time_coordinate(times,c)
    raw_grid=net.decode(grid,s)
    means=raw_grid@data['volumes']/jnp.sum(data['volumes'])
    return fixed_initial(points,c,xp=jnp)[None,:]+s[:,None]*(net.decode(query,s)-means[:,None])


def make_loss(c):
    def one(net,data,times):
        prepared=prepare(net,data)
        def at(t):
            phi,dt=jax.jvp(lambda tt:phase(net,prepared,tt,data,c),(t,),(jnp.ones_like(t),))
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,data)
            residual=c.physical.terminal_time*(dt+c.physical.mobility*apply_operator(mu,data))
            residual=data['metric']@residual
            return jnp.dot(data['volumes'],residual**2)/jnp.sum(data['volumes'])
        if 'initial_residual_fraction' in c.loss:
            times=eqx.error_if(times,times[0]!=0.,'Initial PDE fraction requires a leading physical t=0 node')
        values=jax.vmap(at)(times)
        if 'initial_residual_fraction' in c.loss:
            fraction=c.loss.initial_residual_fraction
            return fraction*values[0]+(1-fraction)*jnp.mean(values[1:])
        return jnp.mean(values)
    def loss(net,batch,times):
        batch=compute_arrays(batch,c)
        times=jnp.asarray(times,dtype=compute_dtype(c))
        return jnp.mean(jax.vmap(one,in_axes=(None,0,None))(net,batch,times))
    return loss


def operator_temporal_samples(c, iteration, key):
    """Keep legacy condition/time draws exact; opt in to the shared time measure."""
    if 'temporal' in c.sample:
        times, horizon = temporal_samples(c, iteration, key)
    else:
        u = jax.random.uniform(key, (c.sample.Nt,), dtype=jnp.float64)
        horizon = c.physical.terminal_time
        times = horizon*(jnp.arange(c.sample.Nt)+u)/c.sample.Nt
    # Keep the FP64 RNG path shared across precisions, then round its result.
    return jnp.concatenate((jnp.zeros(1), times)).astype(compute_dtype(c)), horizon


def train_operator(c):
    validate(c)
    continuation=continuation_spec(c)
    if c.operator.action=='prepare':
        # No JAX backend has been initialized here. FEniCS/meshing runs in its
        # separate interpreter, before any GPU training process.
        with tempfile.TemporaryDirectory(prefix='confined-geometry-') as directory:
            request=Path(directory)/'config.json';request.write_text(c.to_json())
            if 'precompute' in c.operator:
                env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                env.pop('VIRTUAL_ENV',None)
                subprocess.run([sys.executable,'-m','examples.confined_ch2d.operator_precompute',
                    '--request',str(request)],check=True,cwd=Path(__file__).resolve().parents[2],env=env)
            else:
                subprocess.run([c.operator.reference_python,'-m','examples.confined_ch2d.operator_reference',
                                '--request',str(request)],check=True,cwd=Path(__file__).resolve().parents[2])
        return Path(c.operator.dataset_path)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    start=time.perf_counter()
    manifest,refs,bank=load_bank(c)
    monitor,labeled=evaluation_indices(manifest)
    online_enabled='online' in c.operator
    if online_enabled:
        evaluation_bank=jax.tree.map(lambda a:a[jnp.asarray(labeled)],bank)
        evaluation_positions={index:position for position,index in enumerate(labeled)}
    def evaluation_data(index):
        if online_enabled:return jax.tree.map(lambda a:a[evaluation_positions[index]],evaluation_bank)
        return jax.tree.map(lambda a:a[index],bank)
    run_name=c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-geometry-operator'
    run=Path(c.log.log_dir)/run_name;run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(c.to_json(indent=2)+'\n')
    if 'training_metric' in manifest:
        (run/'residual_metric.json').write_text(json.dumps(manifest['training_metric'],indent=2,allow_nan=False)+'\n')
    root=Path(__file__).resolve().parents[2]
    shutil.copytree(Path(__file__).parent,run/'source/examples/confined_ch2d',ignore=shutil.ignore_patterns('__pycache__'))
    for rel in ('src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/nn/conditional.py',
                'src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py'):
        target=run/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
    keys=jax.random.split(jax.random.key(c.train.seed),3)
    net=build_network(c,keys[0]);opt=make_compute_optimizer(c)
    net,state,start_step,continuation=initial_state(net,opt,continuation)
    if continuation is not None:
        (run/'continuation.json').write_text(json.dumps(continuation,indent=2,allow_nan=False)+'\n')
    if continuation is not None or online_enabled:
        eqx.tree_serialise_leaves(run/'optimizer_initial.eqx',state)
    eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    if online_enabled and continuation is None:
        counters=[int(a) for a in jax.tree.leaves(state)
                  if eqx.is_array(a) and a.shape==() and np.issubdtype(a.dtype,np.integer)]
        if start_step!=0 or not counters or any(counters):
            raise ValueError('Cold online training must start with fresh optimizer counters at zero')
        (run/'initial_restore_audit.json').write_text(json.dumps(dict(status='passed',step=0,
            initialization='Fresh network and optimizer; no checkpoint import',
            optimizer_integer_counters=counters,origin_step=c.operator.online.origin_step),indent=2)+'\n')
    if continuation is not None and 'checkpoint' in continuation:
        pairs=((Path(continuation['model']),run/'model_initial.eqx'),
               (Path(continuation['optimizer_state']),run/'optimizer_initial.eqx'))
        exact=[left.read_bytes()==right.read_bytes() for left,right in pairs]
        if not all(exact):raise ValueError('Paired checkpoint did not restore exact model/optimizer arrays')
        (run/'initial_restore_audit.json').write_text(json.dumps(dict(
            status='passed',step=start_step,model_bytes_exact=exact[0],optimizer_bytes_exact=exact[1],
            checkpoint=continuation['checkpoint'],schedule='Restored complete SOAP/Optax state; no reset'),indent=2)+'\n')
    loss=make_loss(c)
    @eqx.filter_jit
    def step(model,state,batch,times):
        value,gradient=eqx.filter_value_and_grad(loss)(model,batch,times)
        finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]))
        update,state=opt.update(gradient,state,model)
        return eqx.apply_updates(model,update),state,value,finite
    predict=eqx.filter_jit(lambda model,data,times:fields(model,data,times,c))
    times_ref=jnp.asarray(refs[labeled[0]]['time'],dtype=compute_dtype(c))
    writer=SummaryWriter(str(run));history=[];timing=[];draws=[];reflection_draws=[];online_draws=[];online_times=[]
    augmented='augmentation' in c.operator
    if augmented:
        from examples.confined_ch2d.operator_augmentation import axis_reflections
        augment=eqx.filter_jit(lambda data,bits:axis_reflections(data,bits,c))
        augmentation_key=jax.random.fold_in(jax.random.key(c.train.seed),103)
    def assess(iteration,final):
        rows=[]
        for index in labeled:
            d=refs[index]
            item=evaluation_data(index)
            values=np.asarray(predict(net,item,times_ref))[:,:len(d['points'])]
            if continuation is not None and iteration==start_step and 'checkpoint' not in continuation:
                with np.load(Path(continuation['parent_run'])/f'prediction_{index:03d}.npz',allow_pickle=False) as parent:
                    for key in ('points','volumes','time','holes'):
                        np.testing.assert_array_equal(d[key],parent[key])
                    np.testing.assert_allclose(values,np.stack((parent['phi'],parent['mu']),axis=-1),rtol=0,atol=1e-9)
            phi,mu=values[...,0],values[...,1]
            mass=phi@d['volumes']/d['volumes'].sum()
            metric=dict(index=index,phi_pooled=relative(phi,d['phi'],d['volumes']),
                phi_terminal=relative(phi[-1],d['phi'][-1],d['volumes']),
                phi_by_time=[relative(a,b,d['volumes']) for a,b in zip(phi,d['phi'],strict=True)],
                mu_pooled=relative(mu,d['mu'],d['volumes']),mu_terminal=relative(mu[-1],d['mu'][-1],d['volumes']),
                mass_max_drift=float(np.max(abs(mass-mass[0]))),phi_min=float(phi.min()),phi_max=float(phi.max()))
            if not np.isfinite(values).all():raise FloatingPointError('Nonfinite geometry operator prediction')
            rows.append(metric)
            if final:
                np.savez_compressed(run/f'prediction_{index:03d}.npz',phi=phi,mu=mu,
                    points=d['points'],volumes=d['volumes'],time=d['time'],holes=d['holes'],
                    mesh_points=d['mesh_points'],mesh_cells=d['mesh_cells'],
                    phi_reference=d['phi'],mu_reference=d['mu'])
        if continuation is not None and iteration==start_step and 'checkpoint' in continuation:
            previous={row['index']:row for row in continuation['parent_evaluation']['conditions']}
            gaps=[]
            for row in rows:
                for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
                    gap=abs(row[key]-previous[row['index']][key]);gaps.append(gap)
                    if gap>2e-5:raise ValueError(f'Initial checkpoint field-score mismatch {row["index"]}/{key}: {gap}')
            (run/'initial_field_audit.json').write_text(json.dumps(dict(status='passed',
                checkpoint_step=start_step,compared_scores=len(gaps),maximum_absolute_score_gap=max(gaps),
                tolerance=2e-5,reason='Fixed zero padding can change FP32 reduction order; physical inputs/model/state identical'),indent=2)+'\n')
        by_index={row['index']:row for row in rows}
        record=dict(step=iteration,elapsed_seconds=time.perf_counter()-start,conditions=rows,
            train_monitor_indices=monitor,test_indices=manifest['test_indices'])
        for name,ids in (('train',monitor),('test',manifest['test_indices'])):
            for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
                value=float(np.mean([by_index[i][key] for i in ids]));record[name+'_'+key]=value
                writer.add_scalar('eval/'+name+'_'+key,value,iteration)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        history.append(record)
        print('EVALUATION',json.dumps({key:value for key,value in record.items() if key!='conditions'}),flush=True)
        return record
    assess(start_step,False)
    setup=time.perf_counter()-start
    pool=None
    if online_enabled:
        from examples.confined_ch2d.operator_online import RollingGeometryPool
        pool=RollingGeometryPool(c,run,start_step,continuation,[refs[index]['holes'] for index in manifest['test_indices']])
    try:
        if pool is not None:bank=pool.start(bank)
        with (run/'loss.jsonl').open('x') as log:
            for i in range(start_step+1,c.train.iterations+1):
                if pool is not None:bank=pool.refresh(bank,i)
                ids=jax.random.randint(jax.random.fold_in(keys[1],i),(c.operator.condition_batch,),0,c.operator.train_conditions)
                batch=jax.tree.map(lambda a:a[ids],bank)
                if pool is not None:online_draws.append(pool.draw_ids[np.asarray(ids)].copy())
                if augmented:
                    bits=jax.random.bernoulli(jax.random.fold_in(augmentation_key,i),shape=(c.operator.condition_batch,2))
                    batch=augment(batch,bits);reflection_draws.append(np.asarray(bits))
                times,horizon=operator_temporal_samples(c,i,jax.random.fold_in(keys[2],i))
                if pool is not None:online_times.append(np.asarray(times))
                tick=time.perf_counter();net,state,value,finite=step(net,state,batch,times)
                value.block_until_ready();timing.append(time.perf_counter()-tick);draws.append(np.asarray(ids))
                if not bool(finite):raise FloatingPointError(f'Nonfinite step/gradient at {i}')
                if i==start_step+1:
                    (run/'memory_first_update.json').write_text(json.dumps(jax.devices()[0].memory_stats(),indent=2,allow_nan=False)+'\n')
                if pool is not None and i==start_step+25:
                    floating=[a for tree in (net,state) for a in jax.tree.leaves(tree) if eqx.is_inexact_array(a)]
                    if not all(np.isfinite(a).all() and a.dtype==compute_dtype(c) for a in floating):
                        raise FloatingPointError('Online production warmup state lost finiteness/compute precision')
                    (run/'online_preflight.json').write_text(json.dumps(dict(status='passed',step=i,
                        warmup_updates=25,mean_last20_update_seconds=float(np.mean(timing[-20:])),
                        all_floating_model_state_finite_and_compute_dtype=True,
                        payload_keys=sorted(batch),
                        memory=jax.devices()[0].memory_stats(),pool=pool.summary()),indent=2,allow_nan=False)+'\n')
                if i==start_step+1 or i%100==0:
                    record=dict(step=i,loss=float(value),training_horizon=float(horizon))
                    if pool is not None:record['online']=pool.summary()
                    log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                    writer.add_scalar('train/physics',float(value),i)
                if i%c.log.eval_every==0 or i==c.train.iterations:assess(i,i==c.train.iterations)
                if i%c.log.checkpoint_every==0:
                    online_state=pool.save_state(i) if pool is not None else None
                    save_checkpoint(run,net,state,i,online_state=online_state)
    finally:
        if pool is not None:pool.close()
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    eqx.tree_serialise_leaves(run/'optimizer_final.eqx',state)
    recorded=dict(condition_ids=np.stack(draws))
    if pool is not None:
        recorded['geometry_draw_ids']=np.stack(online_draws);recorded['times']=np.stack(online_times)
    if augmented:recorded['axis_reflections']=np.stack(reflection_draws)
    np.savez_compressed(run/'training_draws.npz',**recorded)
    # Swap encoder inputs while retaining the physical query/IC/mass/D. The
    # heterogeneous model also encodes the volume points, requiring the encoded
    # context itself to be replaced rather than changing only boundary rows.
    @eqx.filter_jit
    def swapped_fields(model,own,other,times):
        prepared=model.prepare_encoded(encode(model,other),own['points'])
        return jax.vmap(lambda t:phase(model,prepared,t,own,c))(times)
    swaps=[]
    for index in manifest['test_indices']:
        own=evaluation_data(index)
        other=(index+1)%len(refs)
        wrong=evaluation_data(other)
        output=np.asarray(swapped_fields(net,own,wrong,times_ref))[:,:len(refs[index]['points'])]
        swaps.append(dict(index=index,other=other,wrong_condition_phi_pooled=relative(output,refs[index]['phi'],refs[index]['volumes'])))
    summary=dict(wall_seconds=time.perf_counter()-start,setup_seconds=setup,first_update_seconds=timing[0],
        mean_update_seconds_after_100=float(np.mean(timing[100:])) if len(timing)>100 else None,
        final_metrics=history[-1],geometry_swaps=swaps,checkpoint='model_final.eqx',
        checkpoint_selection='Fixed training budget final, no held-out checkpoint selection',
        training=f'Finite{c.operator.train_conditions} geometry-bank PDE-only pilot; no evolved reference values in step payload',
        train_metric_scope=f'{len(monitor)} labeled training monitors; not all {c.operator.train_conditions} training geometries',
        padded_nodes=bank['points'].shape[1],padded_degree=bank['neighbors'].shape[2],
        dataset=c.operator.dataset_path,backend=jax.default_backend())
    summary['start_step']=start_step
    summary['new_updates']=len(timing)
    summary['continuation']=continuation
    summary['cumulative_wall_seconds']=summary['wall_seconds']+(continuation['previous_cumulative_wall_seconds'] if continuation is not None else 0.)
    memory=jax.devices()[0].memory_stats()
    summary['device_memory_stats']=memory
    if pool is not None:
        summary['online']=pool.summary()
        summary['training']='Continuously sampled input-only geometries in a bounded rolling pool; no evolved reference labels'
        summary['train_metric_scope']='Frozen16original training-domain monitors, separate from replaceable pool; not a current-pool aggregate'
    if augmented:
        combinations=np.column_stack((recorded['condition_ids'].ravel(),recorded['axis_reflections'].reshape(-1,2)))
        summary['augmentation']=dict(kind=c.operator.augmentation,distinct_training_geometries=len(np.unique(combinations,axis=0)),
            initial_field='Reevaluate the fixed ambient function at transformed coordinates; never reflect initial values or future labels',
            sampling_time='Geometry augmentation outside parameter AD and synchronized step timing')
        summary['training']=f'Finite{c.operator.train_conditions} base geometries plus axis reflections, PDE-only; no evolved labels'
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    writer.close();print(json.dumps(summary,indent=2),flush=True)
    return run
