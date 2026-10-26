"""Isolated graph-kernel compiler-policy controls; never change training flags."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/local-ad-performance-20261026'


def worker(dtype_name,policy):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy import sparse
    from examples.corrosion2d import residual_experiment as r
    from examples.corrosion2d.configs.train_base import get_config
    from examples.corrosion2d.residuals import graph_action,graph_edge_action
    from examples.corrosion2d.training_precision import cast_floating
    from nonlocal_pinn.operators import build_D
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    jax.config.update('jax_enable_x64',dtype_name=='float64')
    jax.config.update('jax_default_matmul_precision','highest')
    with jax.enable_x64(False):
        c=get_config();_,_,points=r.grid(c)
        op=build_D(jnp.asarray(points,dtype=jnp.float32),c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
        pool=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)
        ids,sub,core,_=pool[0]
        values=jnp.asarray(np.random.default_rng(42).standard_normal((len(ids),c.sample.Nt),dtype=np.float32))
    dtype=np.dtype(dtype_name)
    values,sub=cast_floating((values,sub),dtype)
    row,col,vals,diag=(np.asarray(a) for a in sub)
    n=len(values);v=np.asarray(values,dtype=np.float64)
    off=sparse.csr_matrix((vals.astype(np.float64),(row,col)),shape=(n,n))
    diag_matrix=off+sparse.diags(diag.astype(np.float64))
    edge_matrix=off+sparse.diags(np.bincount(row,weights=-vals.astype(np.float64),minlength=n))
    record={'status':'running','dtype':dtype_name,'policy':policy,'xla_flags':os.environ['XLA_FLAGS'],
            'shapes':{'values':list(values.shape),'edges':len(vals)},'kernels':{}}
    for name,fn,matrix in (('diagonal_offdiagonal',graph_action,diag_matrix),('edge_difference',graph_edge_action,edge_matrix)):
        start=time.perf_counter();f=jax.jit(fn).lower(values,sub).compile();compile_s=time.perf_counter()-start
        for _ in range(5):actual=jax.block_until_ready(f(values,sub))
        timings=[]
        for _ in range(30):
            start=time.perf_counter();actual=jax.block_until_ready(f(values,sub));timings.append(time.perf_counter()-start)
        expected=matrix@v
        error=float(np.linalg.norm(np.asarray(actual,dtype=np.float64)-expected)/np.linalg.norm(expected))
        if error>(2e-6 if dtype_name=='float32' else 1e-12):raise AssertionError(error)
        record['kernels'][name]={'compile_seconds':compile_s,'mean_seconds':float(np.mean(timings)),
                                 'median_seconds':float(np.median(timings)),'samples_seconds':timings,
                                 'relative_error_vs_scipy':error}
    record['status']='completed'
    (OUT/f'operator-{policy}-{dtype_name}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['mean_seconds'] for k,v in record['kernels'].items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=('float32','float64'))
    parser.add_argument('--policy',choices=('deterministic','ordinary'))
    args=parser.parse_args()
    if args.worker:
        if not args.policy:parser.error('worker requires policy')
        worker(args.worker,args.policy);return
    if args.policy:parser.error('policy requires worker')
    completed=json.loads((OUT/'manifest.json').read_text())
    if completed['status']!='completed':raise RuntimeError('Finish main serial GPU study first')
    manifest={'status':'running','maximum_GPU_workers':1,'runs':[]}
    for dtype in ('float32','float64'):
        for policy in ('deterministic','ordinary'):
            env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                     XLA_FLAGS='--xla_gpu_deterministic_ops=true' if policy=='deterministic' else '',
                     JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-local-ad-operator-cache')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',dtype,'--policy',policy]
            log=OUT/f'operator-{policy}-{dtype}.log';start=time.perf_counter()
            with log.open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            manifest['runs'].append({'dtype':dtype,'policy':policy,'command':command,'stdout':str(log),
                                     'returncode':result.returncode,'wall_seconds':time.perf_counter()-start})
            (OUT/'operator-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:raise RuntimeError(log.read_text())
    manifest['status']='completed';(OUT/'operator-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
