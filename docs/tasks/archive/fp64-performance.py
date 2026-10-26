"""Serial, synchronized local GPU precision benchmark; no production changes."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'reports/fp64-performance-20261026'
INITIAL = Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42/model_initial.eqx')


def worker(dtype_name, suite):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from examples.corrosion2d.configs.train_base import get_config
    from examples.corrosion2d import residual_experiment as r
    from examples.corrosion2d.residuals import graph_action, graph_edge_action
    from nonlocal_pinn.operators import build_D
    from nonlocal_pinn.training.runtime import with_run_seed

    jax.config.update('jax_enable_x64', dtype_name == 'float64')
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'gpu': raise RuntimeError('GPU is required, no CPU fallback')
    dtype = np.dtype(dtype_name)
    device = jax.devices()[0]
    if device.platform != 'gpu': raise RuntimeError(device)

    def write(record):
        (OUT/f'{suite}-{dtype_name}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')

    def stats(samples):
        a = np.asarray(samples)
        return dict(mean_seconds=float(a.mean()), median_seconds=float(np.median(a)),
                    min_seconds=float(a.min()), max_seconds=float(a.max()), samples_seconds=a.tolist())

    def memory():
        return {k:int(v) for k,v in device.memory_stats().items() if isinstance(v,(int,np.integer))}

    def cast(tree):
        return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)

    def assert_dtype(tree):
        floats = [a for a in jax.tree.leaves(tree) if eqx.is_inexact_array(a)]
        if not floats or any(np.dtype(a.dtype) != dtype for a in floats):
            raise TypeError([str(a.dtype) for a in floats])
        return dict(leaves=len(floats), bytes=sum(a.size*a.dtype.itemsize for a in floats), dtype=dtype_name)

    def compile_fn(fn, args):
        start = time.perf_counter()
        lowered = fn.lower(*args)
        text = lowered.as_text()
        types = {t:len(re.findall(r'(?:x|<|\s)'+t+r'(?=>|,|\s)',text)) for t in ('f16','bf16','f32','f64')}
        if suite == 'audit':
            (OUT/f'audit-{dtype_name}.stablehlo.mlir').write_text(text)
            return None, time.perf_counter()-start, types
        if types['f32' if dtype_name=='float32' else 'f64'] == 0:
            raise RuntimeError('Expected floating type absent from StableHLO')
        compiled = lowered.compile()
        return compiled, time.perf_counter()-start, types

    def kernel(name, fn, args, repeats):
        assert_dtype(args)
        compiled, compile_s, types = compile_fn(jax.jit(fn), args)
        start = time.perf_counter(); value = jax.block_until_ready(compiled(*args)); first = time.perf_counter()-start
        assert_dtype(value)
        for _ in range(5): jax.block_until_ready(compiled(*args))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter(); value = jax.block_until_ready(compiled(*args))
            samples.append(time.perf_counter()-start)
        item = dict(name=name, compile_seconds=compile_s, first_execution_seconds=first,
                    steady=stats(samples), stablehlo_float_type_tokens=types,
                    output_shape=list(value.shape), output_dtype=str(value.dtype))
        print(json.dumps({k:v for k,v in item.items() if k!='steady'} | {'steady_median_seconds':item['steady']['median_seconds']}),flush=True)
        return item, value

    record = dict(dtype=dtype_name, suite=suite, device=str(device), device_kind=device.device_kind,
                  jax_enable_x64=bool(jax.config.x64_enabled), matmul_precision='highest',
                  versions={p:importlib.metadata.version(p) for p in ('jax','jaxlib','equinox','optax','soap-jax')},
                  memory_at_start=memory(), xla_flags=os.environ['XLA_FLAGS'], status='running')
    write(record)
    if suite in ('micro', 'gemm'):
        record['kernels'] = []
        for m,k,n in ((1024,1024,1024),(4096,4096,4096),(16384,64,64)):
            rng = np.random.default_rng(42)
            ah = rng.standard_normal((m,k), dtype=np.float32)/np.sqrt(np.float32(k))
            bh = rng.standard_normal((k,n), dtype=np.float32)
            a,b = (jax.device_put(v.astype(dtype)) for v in (ah,bh))
            item,value = kernel(f'gemm_{m}_{k}_{n}',lambda x,y:jnp.matmul(x,y,precision=jax.lax.Precision.HIGHEST),(a,b),30)
            expected = ah[0].astype(np.float64)@bh[:,:8].astype(np.float64)
            actual = np.asarray(value[0,:8]).astype(np.float64)
            error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            if error >= (5e-5 if dtype_name=='float32' else 1e-11): raise AssertionError(error)
            item.update(gflops_per_second=2*m*k*n/item['steady']['median_seconds']/1e9,
                        sampled_relative_error_vs_numpy_fp64=error)
            record['kernels'].append(item);write(record)
            del a,b,value;gc.collect()
        if suite == 'gemm':
            record.update(status='completed',memory_at_end=memory());write(record);return
        with jax.enable_x64(False):
            c = with_run_seed(get_config(),42);_,_,points=r.grid(c)
            op=build_D(jnp.asarray(points,dtype=jnp.float32),c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
        sub=cast((op._row_idx,op._col_idx,op._vals,op._diag))
        rng=np.random.default_rng(42);values=jax.device_put(rng.standard_normal((len(points),c.sample.Nt),dtype=np.float32).astype(dtype))
        for name,fn in (('graph_diagonal_plus_offdiagonal',graph_action),('graph_stable_edge_difference',graph_edge_action)):
            item,value=kernel(name,fn,(values,sub),100)
            assert np.all(np.isfinite(np.asarray(value)))
            record['kernels'].append(item);write(record)
    else:
        # Match the production FP32 graph and input values, then promote them.
        # Rebuilding geometry in a different precision would change the experiment.
        with jax.enable_x64(False):
            c=with_run_seed(get_config(),42);_,_,points=r.grid(c)
            x32=jnp.asarray(points,dtype=jnp.float32)
            op=build_D(x32,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
            pool=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)
            zero,one,_=r.boundary_values(c,points)
            x_ic,ic=r.training_initial_data(c,points,points)
            x_zero=jnp.asarray(points[zero],dtype=jnp.float32);x_one=jnp.asarray(points[one],dtype=jnp.float32)
            key_net,key_time=jax.random.split(jax.random.PRNGKey(42))
            template=r.build_model(c,key_net)
            net=eqx.tree_deserialise_leaves(INITIAL,template)
            times32=[r.sample_training_times(c,key_time,i) for i in range(125)]
        with np.load(c.experiment.reference_path) as ref:
            np.testing.assert_array_equal(np.asarray(op._row_idx),ref['graph_row'])
            np.testing.assert_array_equal(-np.asarray(op._vals),ref['graph_weight'])
        x_grid=cast(x32);net=cast(net);x_ic,ic,x_zero,x_one=cast((x_ic,ic,x_zero,x_one))
        times=cast(times32)
        batches=[]
        for ids,sub,core,_ in pool:
            batches.append((x_grid[ids],cast(sub),core & jnp.asarray(~(zero|one))[ids]))
        loss,_=r.make_training_losses(c,r.nondim(c),x_ic,ic,x_zero,x_one)
        optimizer=r.make_tail_optimizer(c);state=optimizer.init(eqx.filter(net,eqx.is_array))
        weights=jnp.ones(5,dtype=dtype);active=jnp.ones(5,dtype=bool)
        record.update(parameter_audit=assert_dtype(net),optimizer_initial_audit=assert_dtype(state),
                      data_audit=assert_dtype((batches,x_ic,ic,x_zero,x_one,times)),
                      config=c.to_dict(),model_initial_path=str(INITIAL),
                      graph_values='Production FP32 coefficients promoted exactly; same graph/discretization',
                      batch_shapes={'nodes':int(batches[0][0].shape[0]),'edges':int(batches[0][1][0].shape[0]),
                                    'times':c.sample.Nt,'IC':len(x_ic),'BC_zero':len(x_zero),'BC_one':len(x_one)})
        write(record)
        step=r.make_step(loss,optimizer,c)
        args=(net,state,weights,times[0],*batches[0],jnp.asarray(True),active)
        compiled,compile_s,types=compile_fn(step,args)
        record.update(compile_seconds=compile_s,stablehlo_float_type_tokens=types);write(record)
        if suite == 'audit':
            record['lowering_seconds']=record.pop('compile_seconds')
            ir=(OUT/f'audit-{dtype_name}.stablehlo.mlir').read_text()
            if dtype_name=='float64' and re.findall(r'tensor<[^>]*xf32>',ir):
                raise TypeError('FP32 non-scalar tensor in FP64 training graph')
            record.update(status='completed',compile_executed=False,
                          dtype_conclusion='Network/PDE/gradient/SOAP tensors follow chosen dtype; Optax LR schedule has FP32 scalars')
            write(record);return
        start=time.perf_counter();result=jax.block_until_ready(compiled(*args));first=time.perf_counter()-start
        net,state,weights,values,finite=result
        if not bool(finite): raise FloatingPointError(values)
        signature=jax.tree.structure(state)
        record.update(first_execution_seconds=first,first_losses=np.asarray(values).tolist(),
                      optimizer_after_first_audit=assert_dtype(state));write(record)
        print(f'{dtype_name} compiled={compile_s:.2f}s first={first:.3f}s',flush=True)
        timings=[]
        for i in range(1,125):
            batch=batches[i%len(batches)]
            start=time.perf_counter()
            result=jax.block_until_ready(compiled(net,state,weights,times[i],*batch,jnp.asarray(i%25==0),active))
            elapsed=time.perf_counter()-start
            net,state,weights,values,finite=result
            if not bool(finite): raise FloatingPointError(f'{i}: {values}')
            if jax.tree.structure(state)!=signature: raise RuntimeError('Optimizer structure changed')
            if i>=50:timings.append(dict(iteration=i,seconds=elapsed,gradnorm_refresh=i%25==0,soap_refresh=i%5==0))
            if (i+1)%25==0: print(f'{dtype_name} completed {i+1}/125, last step {elapsed:.4f}s',flush=True)
        record.update(warmup_updates=50,measured_updates=75,refresh_periods={'gradnorm':25,'soap':5},
                      steady=stats([v['seconds'] for v in timings]),step_samples=timings,
                      block_means_seconds=[float(np.mean([v['seconds'] for v in timings[b:b+25]])) for b in (0,25,50)],
                      ordinary=stats([v['seconds'] for v in timings if not v['soap_refresh']]),
                      soap_only=stats([v['seconds'] for v in timings if v['soap_refresh'] and not v['gradnorm_refresh']]),
                      gradnorm_and_soap=stats([v['seconds'] for v in timings if v['gradnorm_refresh']]),
                      parameter_final_audit=assert_dtype(net),optimizer_final_audit=assert_dtype(state),
                      final_losses=np.asarray(values).tolist(),memory_at_end=memory())
    record.update(status='completed',memory_at_end=memory());write(record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=('float32','float64'))
    parser.add_argument('--suite',choices=('micro','train','gemm','audit'))
    args=parser.parse_args()
    if args.worker is not None:
        if args.suite is None:parser.error('--worker requires --suite')
        worker(args.worker,args.suite);return
    if args.suite is not None:parser.error('--suite requires --worker')
    OUT.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=str(ROOT),
             JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             XLA_FLAGS='--xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-fp64-benchmark-cache')
    manifest=dict(status='running',maximum_GPU_workers=1,runs=[],hardware=subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip())
    for suite in ('micro','train'):
        for dtype in ('float32','float64'):
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',dtype,'--suite',suite]
            log=OUT/f'{suite}-{dtype}.log';start=time.perf_counter()
            print(f'Starting {suite}/{dtype}',flush=True)
            with log.open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            manifest['runs'].append(dict(suite=suite,dtype=dtype,command=command,stdout=str(log),
                                         returncode=result.returncode,wall_seconds=time.perf_counter()-start))
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:raise RuntimeError(log.read_text())
            print(f'Finished {suite}/{dtype}',flush=True)
    manifest['status']='completed';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
