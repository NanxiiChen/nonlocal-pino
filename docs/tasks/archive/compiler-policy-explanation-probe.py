"""Frozen graph-kernel/HLO control; no parameter update or training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/compiler-policy-explanation-20261026'


def worker(policy):
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from examples.corrosion2d.configs.train_base import get_config
    from examples.corrosion2d import residual_experiment as r
    from examples.corrosion2d.residuals import graph_action
    from nonlocal_pinn.operators import build_D
    if jax.default_backend()!='gpu':raise RuntimeError('GPU probe required')
    c=get_config();_,_,points=r.grid(c)
    op=build_D(jnp.asarray(points,dtype=jnp.float32),c.nl.delta,method='sparse',kernel=c.nl.kernel)
    ids,sub,_,_=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)[0]
    values=jnp.asarray(np.random.default_rng(42).standard_normal((len(ids),16),dtype=np.float32))
    forward=jax.jit(graph_action).lower(values,sub).compile()
    gradient=jax.jit(jax.grad(lambda v,s:jnp.mean(graph_action(v,s)**2))).lower(values,sub).compile()
    text=forward.as_text();(OUT/f'{policy}-forward.hlo').write_text(text)
    results={}
    arrays={}
    for name,fn in [('forward',forward),('gradient',gradient)]:
        seq=[np.asarray(jax.block_until_ready(fn(values,sub))).copy() for _ in range(10)]
        delta=max(float(np.max(np.abs(a-seq[0]))) for a in seq)
        results[name]={'repeat_max_abs_difference':delta,'repeat_max_relative_difference':max(float(np.linalg.norm(a-seq[0])/np.linalg.norm(seq[0])) for a in seq)}
        arrays[name]=seq[0]
    np.savez_compressed(OUT/f'{policy}.npz',values=np.asarray(values),**arrays)
    record={'policy':policy,'xla_flags':os.environ['XLA_FLAGS'],'jax':jax.__version__,'jaxlib':jaxlib.__version__,
            'values_shape':list(values.shape),'edges':len(sub[0]),'repeats':10,'results':results,
            'optimized_hlo_control_ops':[line.strip() for line in text.splitlines() if '= ' in line and (' while(' in line or ' scatter(' in line)]}
    (OUT/f'{policy}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('deterministic','ordinary'));args=parser.parse_args()
    if args.worker:worker(args.worker);return
    OUT.mkdir(parents=True,exist_ok=False)
    for policy in ('deterministic','ordinary'):
        env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                 XLA_FLAGS='--xla_gpu_deterministic_ops=true' if policy=='deterministic' else '')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',policy],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
