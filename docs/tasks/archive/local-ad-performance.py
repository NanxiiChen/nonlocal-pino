"""Serial GPU throughput controls: current corrosion graph versus local AD."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT = ROOT/'reports/local-ad-performance-20261026'
INITIAL = Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42/model_initial.eqx')
KINDS = ('graph', 'local_tensor', 'local_joint')


def spatial_laplacians(net, x, t, cle):
    """Forward-over-reverse spatial Hessians of phi and stable KKS q."""
    import jax
    import jax.numpy as jnp
    def spatial_fields(xx):
        raw = net.net(xx, t)
        physical = net.constraint(raw)
        q = .5*cle*(jnp.tanh(raw[1])-1)
        return jnp.stack([physical[0], q])

    hess = jax.hessian(spatial_fields)(x)
    return jnp.trace(hess[0]), jnp.trace(hess[1])


def local_point(net, x, t, cle):
    """Two scalar time grads plus spatial Hessians, no FD."""
    import jax
    fields = net(x, t)
    pt = jax.grad(lambda tt: net(x, tt)[0])(t)
    ct = jax.grad(lambda tt: net(x, tt)[1])(t)
    lap_p,lap_q=spatial_laplacians(net,x,t,cle)
    return fields[0], fields[1], pt, ct, lap_p, lap_q


def local_residual(net, times, points, nd, kind):
    """Local PDE agrees with the existing D_delta -> -Laplacian branch."""
    import jax
    import jax.numpy as jnp
    n_ch, n1, n2, n3, dc, cle = nd
    if kind == 'local_tensor':
        def values(ts):
            return jax.vmap(lambda t:jax.vmap(net,in_axes=(0,None))(points,t))(ts)
        fields,dt=jax.jvp(values,(times,),(jnp.ones_like(times),))
        lap_p,lap_q=jax.vmap(lambda t:jax.vmap(lambda x:spatial_laplacians(net,x,t,cle))(points))(times)
        data=fields[...,0],fields[...,1],dt[...,0],dt[...,1],lap_p,lap_q
    elif kind == 'local_joint':
        data = jax.vmap(lambda xt: local_point(net,xt[:2],xt[2],cle))(points)
    else:
        raise ValueError(kind)
    p, c, pt, ct, lap_p, lap_q = data
    q = c-dc*(3*p**2-2*p**3)-cle
    ac = pt-n3*lap_p+n2*(4*p**3-6*p**2+2*p)-n1*q*dc*(-6*p**2+6*p)
    ch = ct-n_ch*lap_q
    # All loss reductions use (space,time), matching the production graph path.
    return (ac.T, ch.T) if kind=='local_tensor' else (ac, ch)


def local_loss_factory(config, nd, x_ic, ic, x_zero, x_one, kind):
    import jax
    import jax.numpy as jnp
    from examples.corrosion2d.residuals import normalized_data_fields

    def pde_losses(net, times, points, sub, mask):
        ac, ch = local_residual(net,times,points,nd,kind)
        weights = mask.astype(ac.dtype)[:,None] if kind=='local_tensor' else mask.astype(ac.dtype)
        count = jnp.sum(weights)*(len(times) if kind=='local_tensor' else 1)
        return jnp.stack([config.loss.ac_scale*jnp.sum(jnp.abs(ac)*weights)/count,
                          config.loss.ch_scale*jnp.sum(jnp.abs(ch)*weights)/count])

    def loss_vector(net, times, points, sub, mask):
        pde = pde_losses(net,times,points,sub,mask)
        zero = normalized_data_fields(net,x_zero,times)
        one = normalized_data_fields(net,x_one,times)
        bc_zero = jnp.mean(jnp.sum(zero**2,axis=-1))
        bc_one = jnp.mean(jnp.sum((one-1)**2,axis=-1))
        initial = jax.vmap(net,in_axes=(0,None))(x_ic,jnp.array(0.,dtype=times.dtype))
        ic_loss = jnp.mean(jnp.sum((initial-ic)**2,axis=-1))
        return jnp.concatenate([pde,jnp.stack([bc_zero,bc_one,ic_loss])])
    return loss_vector, pde_losses


def stats(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    return {'mean_seconds':float(a.mean()),'median_seconds':float(np.median(a)),
            'min_seconds':float(a.min()),'max_seconds':float(a.max()),
            'samples_seconds':a.tolist()}


def worker(kind, dtype_name, policy):
    import importlib.metadata
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from examples.corrosion2d import residual_experiment as r
    from examples.corrosion2d.configs.train_base import get_config
    from examples.corrosion2d.residuals import compute_transport_pde
    from examples.corrosion2d.training_precision import cast_floating, audit_floating
    from nonlocal_pinn.operators import build_D
    from nonlocal_pinn.training.runtime import with_run_seed

    dtype=np.dtype(dtype_name)
    jax.config.update('jax_enable_x64',dtype_name=='float64')
    jax.config.update('jax_default_matmul_precision','highest')
    if jax.default_backend()!='gpu': raise RuntimeError('GPU required; no CPU fallback')
    device=jax.devices()[0]
    with jax.enable_x64(False):
        c=with_run_seed(get_config(),42)
        c.train.compute_dtype=dtype_name
        _,_,points=r.grid(c)
        x32=jnp.asarray(points,dtype=jnp.float32)
        op=build_D(x32,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
        pool=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)
        zero,one,_=r.boundary_values(c,points)
        xic,ic=r.training_initial_data(c,points,points)
        xzero=jnp.asarray(points[zero],dtype=jnp.float32)
        xone=jnp.asarray(points[one],dtype=jnp.float32)
        kn,kt=jax.random.split(jax.random.PRNGKey(42))
        template=r.build_model(c,kn)
        initial=eqx.tree_deserialise_leaves(INITIAL,template)
        for a,b in zip(jax.tree.leaves(eqx.filter(initial,eqx.is_array)),
                       jax.tree.leaves(eqx.filter(template,eqx.is_array)),strict=True):
            np.testing.assert_array_equal(a,b)
        batches=[];counts=[]
        for ids,sub,core,_ in pool:
            valid=np.asarray(core)&(~(zero|one))[np.asarray(ids)]
            count=int(valid.sum());counts.append(count)
            if kind=='graph':
                batches.append((x32[ids],sub,jnp.asarray(valid)))
            else:
                selected=np.asarray(ids)[np.flatnonzero(valid)]
                selected=np.pad(selected,(0,c.sample.K_core-count),mode='edge')
                mask=jnp.arange(c.sample.K_core)<count
                if kind=='local_tensor': batches.append((x32[selected],None,mask))
                else: batches.append((jnp.zeros((c.sample.K_core*c.sample.Nt,3),dtype=jnp.float32),None,jnp.tile(mask,c.sample.Nt)))
    with np.load(c.experiment.reference_path) as ref:
        for a,key in ((op._row_idx,'graph_row'),(op._col_idx,'graph_col'),(-op._vals,'graph_weight')):
            np.testing.assert_array_equal(a,ref[key])
    initial,xic,ic,xzero,xone,batches=cast_floating((initial,xic,ic,xzero,xone,batches),dtype)
    nd=r.nondim(c)
    if kind=='graph':
        loss,_=r.make_training_losses(c,nd,xic,ic,xzero,xone)

        def pde_losses(n,ts,x,sub,mask):
            ac,ch=compute_transport_pde(n,x,ts,sub,nd,'phase',c.experiment.latent_normalization,
                                       transport_evaluation='kks_raw_edges')
            weight=mask.astype(ac.dtype)[:,None]
            count=jnp.sum(weight)*len(ts)
            return jnp.stack([c.loss.ac_scale*jnp.sum(jnp.abs(ac)*weight)/count,
                              c.loss.ch_scale*jnp.sum(jnp.abs(ch)*weight)/count])
    else:
        loss,pde_losses=local_loss_factory(c,nd,xic,ic,xzero,xone,kind)
    optimizer=r.make_tail_optimizer(c)
    state=optimizer.init(eqx.filter(initial,eqx.is_array))
    step=r.make_step(loss,optimizer,c)
    active=jnp.ones(5,dtype=bool);weights=jnp.ones(5,dtype=dtype)

    def sample(i):
        batch=batches[i%len(batches)]
        with jax.enable_x64(False):
            ts=r.sample_training_times(c,kt,i)
            if kind=='local_joint':
                xt=jax.random.uniform(jax.random.fold_in(kt,100000+i),(c.sample.K_core*c.sample.Nt,3),dtype=jnp.float32)
                xt=xt*jnp.asarray([1.,.5,1.-c.sample.t_min],dtype=jnp.float32)
                xt=xt+jnp.asarray([0.,0.,c.sample.t_min],dtype=jnp.float32)
        if kind=='local_joint': batch=(xt.astype(dtype),batch[1],batch[2])
        return ts.astype(dtype),batch

    record={'status':'running','kind':kind,'dtype':dtype_name,'policy':policy,'device':device.device_kind,
            'versions':{p:importlib.metadata.version(p) for p in ('jax','jaxlib','equinox','soap-jax')},
            'config':c.to_dict(),'initial_path':str(INITIAL),'paired_initial_exact':True,
            'precision_policy':'Common FP32 initial/data/operator/time/random-space-time draws promoted exactly',
            'xla_flags':os.environ['XLA_FLAGS'],'parameter_dtype':audit_floating(initial,dtype),
            'optimizer_initial_dtype':audit_floating(state,dtype),
            'active_spatial_core_count_min_max':[min(counts),max(counts)],
            'active_PDE_pairs_per_batch_min_max':[min(counts)*c.sample.Nt,max(counts)*c.sample.Nt],
            'nominal_PDE_pairs':c.sample.K_core*c.sample.Nt,
            'network_PDE_points':len(batches[0][0])*c.sample.Nt if kind!='local_joint' else len(batches[0][0]),
            'batch_shapes':{'points':list(batches[0][0].shape),'times':c.sample.Nt,'IC':len(xic),'BC':[len(xzero),len(xone)],
                            'graph_padded_edges':len(batches[0][1][0]) if kind=='graph' else 0},
            'sampling_in_update_timing':True,'reference_accuracy_evaluated':False}
    name=f'{kind}-{dtype_name}' if policy=='deterministic' else f'{kind}-ordinary-{dtype_name}'
    path=OUT/f'{name}.json'
    def write():path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    write()
    ts,batch=sample(0)
    args=(initial,state,weights,ts,*batch,jnp.asarray(True),active)
    start=time.perf_counter();compiled=step.lower(*args).compile()
    record['training_compile_seconds']=time.perf_counter()-start;write()
    start=time.perf_counter()
    net,state,weights,losses,finite=jax.block_until_ready(compiled(*args))
    if not bool(finite):raise FloatingPointError(losses)
    record['first_execution_seconds']=time.perf_counter()-start
    record['first_losses']=np.asarray(losses).tolist();write()
    print(f'{kind}/{dtype_name} compiled={record["training_compile_seconds"]:.1f}s',flush=True)
    samples=[]
    for i in range(1,125):
        start=time.perf_counter()
        ts,batch=sample(i)
        net,state,weights,losses,finite=jax.block_until_ready(compiled(net,state,weights,ts,*batch,jnp.asarray(i%25==0),active))
        elapsed=time.perf_counter()-start
        if not bool(finite):raise FloatingPointError(f'{i}: {losses}')
        if i>=50:samples.append({'iteration':i,'seconds':elapsed,'soap_refresh':i%5==0,'gradnorm_refresh':i%25==0})
        if (i+1)%25==0:print(f'{kind}/{dtype_name} {i+1}/125, last={elapsed:.4f}s',flush=True)
    record.update(warmup_updates=50,measured_updates=75,update_samples=samples,
                  steady=stats([v['seconds'] for v in samples]),
                  ordinary=stats([v['seconds'] for v in samples if not v['soap_refresh']]),
                  soap_only=stats([v['seconds'] for v in samples if v['soap_refresh'] and not v['gradnorm_refresh']]),
                  gradnorm_soap=stats([v['seconds'] for v in samples if v['gradnorm_refresh']]),
                  block_means_seconds=[float(np.mean([v['seconds'] for v in samples[b:b+25]])) for b in (0,25,50)],
                  parameters_final_dtype=audit_floating(net,dtype),optimizer_final_dtype=audit_floating(state,dtype),
                  last_losses=np.asarray(losses).tolist(),
                  training_memory={k:int(v) for k,v in device.memory_stats().items() if isinstance(v,(int,np.integer))})
    write()
    # Frozen initial-model PDE-only cost, with sampling and all device work synced.
    ts,batch=sample(0)
    start=time.perf_counter();pde=eqx.filter_jit(pde_losses).lower(initial,ts,*batch).compile()
    record['pde_compile_seconds']=time.perf_counter()-start
    for i in range(5):
        ts,batch=sample(i);jax.block_until_ready(pde(initial,ts,*batch))
    times=[]
    for i in range(30):
        start=time.perf_counter();ts,batch=sample(i)
        value=jax.block_until_ready(pde(initial,ts,*batch));times.append(time.perf_counter()-start)
        if not np.isfinite(np.asarray(value)).all():raise FloatingPointError(value)
    record['pde_only']=stats(times);write()
    if kind=='local_joint':
        time_controls={}
        def grad_time(n,xt):
            return jax.vmap(lambda z:jnp.stack([jax.grad(lambda tt:n(z[:2],tt)[k])(z[2]) for k in range(2)]))(xt)
        def jvp_time(n,xt):
            def values(ts):return jax.vmap(lambda x,t:n(x,t))(xt[:,:2],ts)
            return jax.jvp(values,(xt[:,2],),(jnp.ones_like(xt[:,2]),))[1]
        values=[]
        for label,fn in (('grad',grad_time),('jvp',jvp_time)):
            start=time.perf_counter();f=eqx.filter_jit(fn).lower(initial,batch[0]).compile()
            compile_s=time.perf_counter()-start
            for _ in range(5):value=jax.block_until_ready(f(initial,batch[0]))
            timings=[]
            for _ in range(30):
                start=time.perf_counter();value=jax.block_until_ready(f(initial,batch[0]));timings.append(time.perf_counter()-start)
            values.append(np.asarray(value))
            time_controls[label]={'compile_seconds':compile_s,'steady':stats(timings)}
        error=float(np.max(np.abs(values[0]-values[1])))
        np.testing.assert_allclose(values[0],values[1],atol=2e-6 if dtype_name=='float32' else 1e-12,rtol=1e-4 if dtype_name=='float32' else 1e-10)
        record['joint_time_only_control']={'grad_vs_jvp':time_controls,'max_abs_difference':error,'points':len(batch[0]),'sampling_timed':False}
    record['status']='completed';write()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=KINDS)
    parser.add_argument('--dtype',choices=('float32','float64'))
    parser.add_argument('--policy',choices=('deterministic','ordinary'),default='deterministic')
    args=parser.parse_args()
    if args.worker is not None:
        if args.dtype is None:parser.error('--worker requires --dtype')
        if args.policy=='ordinary' and args.worker!='graph':parser.error('ordinary control is graph-only')
        worker(args.worker,args.dtype,args.policy);return
    if args.dtype is not None:parser.error('--dtype requires --worker')
    OUT.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
             XLA_FLAGS='--xla_gpu_deterministic_ops=true',
             JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-local-ad-performance-cache')
    manifest={'status':'running','maximum_GPU_workers':1,'runs':[],
              'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()}
    for dtype in ('float32','float64'):
        for kind in KINDS:
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',kind,'--dtype',dtype]
            log=OUT/f'{kind}-{dtype}.log';start=time.perf_counter()
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(f'Start {kind}/{dtype}',flush=True)
            with log.open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            manifest['runs'].append({'kind':kind,'dtype':dtype,'command':command,'stdout':str(log),
                                     'returncode':result.returncode,'wall_seconds':time.perf_counter()-start})
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:raise RuntimeError(log.read_text()[-7000:])
            print(f'Finished {kind}/{dtype}',flush=True)
    manifest['status']='completed';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
