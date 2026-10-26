"""Actual electropolish/melting steps, paired GPU policies, serial workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/phasefield-gpu-policy-20261026'


def stats(values):
    import numpy as np
    a=np.asarray(values)
    return {'mean_seconds':float(np.mean(a)),'median_seconds':float(np.median(a)),
            'min_seconds':float(np.min(a)),'max_seconds':float(np.max(a)),
            'samples_seconds':a.tolist()}


def prepare(case,policy):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nonlocal_pinn.operators import build_D
    from nonlocal_pinn.training.runtime import with_run_seed
    if case=='electropolish2d':
        from examples.electropolish2d.configs.train_base import get_config
        from examples.corrosion2d import residual_experiment as r
        from examples.corrosion2d.residuals import graph_action,compute_transport_pde
        c=with_run_seed(get_config(),42);c.train.gpu_policy=policy
        _,_,points=r.grid(c);grid=jnp.asarray(points,dtype=jnp.float32)
        op=build_D(grid,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
        zero,one,_=r.boundary_values(c,points)
        xr0,xr1=r.electropolish_boundary_points(c);xi=r.electropolish_initial_points(c)
        x0,x1=r.training_boundary_data(c,points,xr0,xr1);xi,ic=r.training_initial_data(c,points,xi)
        kn,kt=jax.random.split(jax.random.PRNGKey(42));net=r.build_model(c,kn)
        pool=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)
        batches=[(grid[ids],sub,core&jnp.asarray(~(zero|one))[ids]) for ids,sub,core,_ in pool]
        loss,_=r.make_training_losses(c,r.nondim(c),xi,ic,x0,x1)
        optimizer=r.make_tail_optimizer(c);step=r.make_step(loss,optimizer,c)
        def sample(i):return r.sample_training_times(c,kt,i)
        def args(n,state,w,ts,batch,i):return (n,state,w,ts,*batch,jnp.asarray(i%c.train.gradnorm_update_every==0),jnp.ones(5,dtype=bool))
        def unpack(result):
            n,state,w,losses,finite=result
            return n,state,w,losses,finite
        def pde(n,ts,x,sub,mask):
            ac,ch=compute_transport_pde(n,x,ts,sub,r.nondim(c),'phase',c.experiment.latent_normalization,transport_evaluation=c.experiment.transport_evaluation)
            m=mask.astype(ac.dtype)[:,None];count=jnp.sum(m)*len(ts)
            return jnp.stack([c.loss.ac_scale*jnp.sum(jnp.abs(ac)*m)/count,c.loss.ch_scale*jnp.sum(jnp.abs(ch)*m)/count])
        kernel=graph_action
        data={'ic_points':np.asarray(xi),'ic':np.asarray(ic),'bc0':np.asarray(x0),'bc1':np.asarray(x1)}
        weights=jnp.ones(5,dtype=jnp.float32)
    elif case=='icemelting3d':
        from examples.icemelting3d.configs.train_base import get_config
        from examples.icemelting3d import train as train_mod
        from examples.icemelting3d import losses
        from nonlocal_pinn.training.optimizers import make_optimizer
        c=with_run_seed(get_config(),42);train_mod._validate_config(c)
        axes=[jnp.linspace(0.,1.,n,dtype=jnp.float32) for n in (c.sample.Nx,c.sample.Ny,c.sample.Nz)]
        xx,yy,zz=jnp.meshgrid(*axes,indexing='xy');grid=jnp.stack([xx.ravel(),yy.ravel(),zz.ravel()],-1)
        kn,_=jax.random.split(jax.random.PRNGKey(42));net=train_mod._build_phase_network(c,kn)
        op=build_D(grid,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
        raw=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)
        batches=[(grid[ids],sub,core) for ids,sub,core,_ in raw]
        nd=train_mod._nondim_params(c);xi=grid[:0];ic=jnp.empty((0,),dtype=grid.dtype)
        lfn=losses.make_loss_fn(grid,op,*nd[:3],x_ic=xi,alpha=c.train.alpha_gradnorm,gradnorm_cap=c.loss.gradnorm_weight_cap,causal=None,formulation=c.phase.formulation)
        optimizer=make_optimizer(c);step=train_mod._make_step(lfn,optimizer)
        bias=jnp.asarray(c.loss.gradnorm_bias,dtype=jnp.float32);gamma=jnp.asarray(c.loss.iface_gamma_decay_to,dtype=jnp.float32)
        def sample(i):return jax.random.uniform(jax.random.PRNGKey(i*3+1000),(c.sample.Nt,),minval=c.sample.t_min,maxval=1.)
        def args(n,state,w,ts,batch,i):return (n,state,ts,ic,w,bias,True,gamma,None,*batch)
        def unpack(result):
            n,state,total,aux,w=result
            return n,state,w,jnp.stack([aux['loss/pde'],aux['loss/ic']]),aux['finite_update']
        def pde(n,ts,x,sub,mask):
            return losses.loss_pde(n,x,ts,sub,mask,*nd[:3],iface_gamma=gamma,causal=None,eps=None,formulation=c.phase.formulation)[0]
        def kernel(values,sub):
            row,col,vals,_=sub
            ratio=losses._tanh_difference_ratio(values[row],values[col])
            return jax.ops.segment_sum(-vals[:,None]*ratio,row,num_segments=len(values))
        data={'ic_points':np.asarray(xi),'ic':np.asarray(ic)};weights=jnp.ones(2,dtype=jnp.float32)
    else:raise ValueError(case)
    return c,net,op,batches,optimizer,step,sample,args,unpack,pde,kernel,weights,data


def kernels(case,kernel,sub,points,nt,directory):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy import sparse
    row,col,vals,diag=(np.asarray(v) for v in sub);n=len(points)
    random=np.random.default_rng(42)
    if case=='icemelting3d':
        radius=.35;width=np.sqrt(2.)*.0225
        u0=(radius-np.linalg.norm(np.asarray(points,dtype=np.float64)-.5,axis=-1))/width
        values=jnp.asarray(u0[:,None]+random.normal(0,.05,(n,nt)),dtype=jnp.float32)
    else:values=jnp.asarray(random.standard_normal((n,nt),dtype=np.float32))
    v=np.asarray(values,dtype=np.float64)
    if case=='electropolish2d':
        matrix=sparse.csr_matrix((vals.astype(np.float64),(row,col)),shape=(n,n))+sparse.diags(diag.astype(np.float64))
        expected=matrix@v;grad_expected=matrix.T@(2*expected/expected.size)
    else:
        w=-vals.astype(np.float64);ui,uj=v[row],v[col]
        logratio=np.logaddexp(ui,-ui)-np.logaddexp(uj,-uj)
        ratio=np.sinh(ui-uj)*np.exp(logratio)
        expected=np.zeros_like(v);np.add.at(expected,row,w[:,None]*ratio)
        cotangent=2*expected/expected.size
        grad_expected=np.zeros_like(v)
        np.add.at(grad_expected,row,w[:,None]*(1+2*np.tanh(ui)*ratio)*cotangent[row])
        np.add.at(grad_expected,col,-w[:,None]*np.exp(2*logratio)*cotangent[row])
    record={};arrays={'values':np.asarray(values)}
    for name,fn,reference in [('forward',kernel,expected),('input_gradient',jax.grad(lambda v,s:jnp.mean(kernel(v,s)**2)),grad_expected)]:
        start=time.perf_counter();compiled=jax.jit(fn).lower(values,sub).compile();compile_s=time.perf_counter()-start
        hlo=compiled.as_text();(directory/f'{name}.hlo').write_text(hlo)
        repeats=[]
        for _ in range(5):repeats.append(np.asarray(jax.block_until_ready(compiled(values,sub))).copy())
        timings=[]
        for _ in range(30):
            start=time.perf_counter();actual=jax.block_until_ready(compiled(values,sub));timings.append(time.perf_counter()-start)
        output=np.asarray(actual,dtype=np.float64);error=float(np.linalg.norm(output-reference)/np.linalg.norm(reference))
        tolerance=2e-6 if case=='electropolish2d' else 2e-5
        if not np.isfinite(output).all() or error>=tolerance:raise AssertionError((case,name,error,tolerance))
        record[name]={'compile_seconds':compile_s,'steady':stats(timings),'relative_error_vs_independent_numpy_scipy':error,
                      'repeat_max_relative_difference':max(float(np.linalg.norm(a-repeats[0])/np.linalg.norm(repeats[0])) for a in repeats),
                      'optimized_hlo_control_ops':[line.strip() for line in hlo.splitlines() if '= ' in line and (' while(' in line or ' scatter(' in line)]}
        arrays[name]=output
    np.savez_compressed(directory/'kernel-arrays.npz',**arrays)
    return record


def worker(case,policy):
    import importlib.metadata
    import equinox as eqx
    import jax
    import numpy as np
    from examples.corrosion2d.training_precision import audit_floating
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required, no fallback')
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision','highest')
    directory=OUT/f'{case}-{policy}';directory.mkdir(exist_ok=False)
    print(f'Preparing actual {case} default grid/pool/model',flush=True)
    start=time.perf_counter()
    c,initial,op,batches,optimizer,step,sample,args,unpack,pde,kernel,weights,data=prepare(case,policy)
    prep_s=time.perf_counter()-start
    state=optimizer.init(eqx.filter(initial,eqx.is_array))
    eqx.tree_serialise_leaves(directory/'initial.eqx',initial)
    np.savez_compressed(directory/'initial-arrays.npz',**{f'leaf_{i}':np.asarray(a) for i,a in enumerate(jax.tree.leaves(eqx.filter(initial,eqx.is_array)))})
    pool={'points':np.stack([np.asarray(v[0]) for v in batches]),'mask':np.stack([np.asarray(v[2]) for v in batches]),
          **{key:np.stack([np.asarray(v[1][i]) for v in batches]) for i,key in enumerate(('row','col','vals','diag'))},
          'times':np.stack([np.asarray(sample(i)) for i in range(125)]),**data}
    np.savez_compressed(directory/'paired-inputs.npz',**pool);del pool
    record={'status':'running','case':case,'policy':policy,'dtype':'float32','config':c.to_dict(),
            'device':jax.devices()[0].device_kind,'versions':{k:importlib.metadata.version(k) for k in ('jax','jaxlib','equinox','soap-jax')},
            'xla_flags':os.environ['XLA_FLAGS'],'preparation_seconds':prep_s,'seed_as_run_provenance':42,
            'grid_nodes':op._N,'global_graph_edges':len(op._vals),'pool_size':len(batches),
            'batch':{'support_nodes':len(batches[0][0]),'padded_edges':len(batches[0][1][0]),'Nt':c.sample.Nt,
                     'valid_core_min_max':[int(min(np.asarray(v[2]).sum() for v in batches)),int(max(np.asarray(v[2]).sum() for v in batches))]},
            'parameter_dtype':audit_floating(initial,np.dtype('float32')),'optimizer_dtype':audit_floating(state,np.dtype('float32')),
            'protocol':{'total_updates':125,'warmup_updates':50,'measured_updates':75,'sampling_timed':True,'device_sync_each_update':True,'logging_evaluation_save_timed':False},
            'accuracy_training_claim':False}
    path=directory/'result.json'
    def write():path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    write();ts=sample(0);batch=batches[0]
    firstargs=args(initial,state,weights,ts,batch,0)
    start=time.perf_counter();compiled=step.lower(*firstargs).compile();record['training_compile_seconds']=time.perf_counter()-start;write()
    start=time.perf_counter();result=jax.block_until_ready(compiled(*firstargs));record['first_execution_seconds']=time.perf_counter()-start
    net,state,weights,losses,finite=unpack(result)
    if not bool(finite):raise FloatingPointError(losses)
    record['initial_losses']=np.asarray(losses).tolist();write()
    print(f'{case}/{policy}: compiled {record["training_compile_seconds"]:.1f}s',flush=True)
    samples=[]
    for i in range(1,125):
        start=time.perf_counter();ts=sample(i);batch=batches[i%len(batches)]
        result=jax.block_until_ready(compiled(*args(net,state,weights,ts,batch,i)))
        elapsed=time.perf_counter()-start;net,state,weights,losses,finite=unpack(result)
        if not bool(finite):raise FloatingPointError((i,losses))
        if i>=50:samples.append({'iteration':i,'seconds':elapsed,'soap_refresh':i%c.train.soap_precond_freq==0,
                                 'gradnorm_refresh':case=='electropolish2d' and i%c.train.gradnorm_update_every==0})
        if (i+1)%25==0:print(f'{case}/{policy}: {i+1}/125, last={elapsed:.5f}s',flush=True)
    record.update(samples=samples,steady=stats([v['seconds'] for v in samples]),
                  block_means_seconds=[float(np.mean([v['seconds'] for v in samples[i:i+25]])) for i in (0,25,50)],
                  parameters_final_dtype=audit_floating(net,np.dtype('float32')),optimizer_final_dtype=audit_floating(state,np.dtype('float32')),
                  final_losses=np.asarray(losses).tolist())
    for category,predicate in [('plain',lambda v:not v['soap_refresh']),('soap',lambda v:v['soap_refresh'] and not v['gradnorm_refresh']),('gradnorm_soap',lambda v:v['gradnorm_refresh'])]:
        group=[v['seconds'] for v in samples if predicate(v)]
        if group:record[category]=stats(group)
    write();print(f'{case}/{policy}: frozen kernels',flush=True)
    record['kernels']=kernels(case,kernel,batches[0][1],batches[0][0],c.sample.Nt,directory);write()
    ts=sample(0);batch=batches[0];start=time.perf_counter();f=eqx.filter_jit(pde).lower(initial,ts,*batch).compile();record['pde_compile_seconds']=time.perf_counter()-start
    for _ in range(5):jax.block_until_ready(f(initial,ts,*batch))
    timings=[]
    for _ in range(30):
        start=time.perf_counter();value=jax.block_until_ready(f(initial,ts,*batch));timings.append(time.perf_counter()-start)
    if not np.isfinite(np.asarray(value)).all():raise FloatingPointError(value)
    record['pde_only']=stats(timings)
    record['memory_stats']={k:int(v) for k,v in jax.devices()[0].memory_stats().items() if isinstance(v,(int,np.integer))}
    record['status']='completed';write()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('electropolish2d','icemelting3d'));parser.add_argument('--policy',choices=('ordinary','deterministic'));args=parser.parse_args()
    if args.case:
        if not args.policy:parser.error('--policy required')
        worker(args.case,args.policy);return
    if args.policy:parser.error('--case required')
    OUT.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','maximum_GPU_workers':1,'runs':[]}
    for case in ('electropolish2d','icemelting3d'):
        for policy in ('ordinary','deterministic'):
            env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                     XLA_FLAGS='--xla_gpu_deterministic_ops=true' if policy=='deterministic' else '',
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-other-case-policy-cache')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--case',case,'--policy',policy]
            log=OUT/f'{case}-{policy}.log';start=time.perf_counter();item={'case':case,'policy':policy,'command':command,'stdout':str(log),'status':'running'}
            manifest['runs'].append(item);(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(f'Start {case}/{policy}',flush=True)
            with log.open('x') as stream:result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            item.update(returncode=result.returncode,wall_seconds=time.perf_counter()-start,status='completed' if result.returncode==0 else 'failed')
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:raise RuntimeError(log.read_text()[-7000:])
    manifest['status']='completed';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
