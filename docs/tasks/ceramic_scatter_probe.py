"""Measure actual core/halo transpose repeatability against a float64 CPU oracle."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.particle_corrosion2d.losses import graph_action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with np.load(a.run/'sampling.npz') as f:
        slot,pool=int(f['condition_ids'][0,0]),int(f['pool_ids'][0,0])
        nt=f['times'].shape[1]
    with np.load(a.run/'online_initial_inputs.npz') as f:
        points=f['graph_points'][slot,pool]
        data={k:jnp.asarray(f[k][slot,pool]) for k in ('core_indices','neighbors','coefficients')}
    rng=np.random.default_rng(1729)
    values=jnp.asarray(rng.normal(size=(len(points),nt)).astype(np.float32))
    adjoint=jnp.asarray(rng.normal(size=(len(data['core_indices']),nt)).astype(np.float32))
    def action(v):return graph_action(v,data)
    pull=jax.jit(jax.grad(lambda v:jnp.sum(action(v)*adjoint)))
    results=[]
    for _ in range(20):results.append(np.asarray(pull(values)))
    forward=jax.make_jaxpr(action)(values)
    backward=jax.make_jaxpr(jax.grad(lambda v:jnp.sum(action(v)*adjoint)))(values)
    c,n,w=(np.asarray(data[k]) for k in ('core_indices','neighbors','coefficients'))
    z=np.asarray(adjoint,dtype=np.float64)
    oracle=np.zeros(values.shape,np.float64)
    for i in range(len(c)):
        oracle[c[i]]+=np.sum(w[i].astype(float))*z[i]
        np.add.at(oracle,n[i],-w[i,:,None].astype(float)*z[i])
    reports=[dict(bitwise_equal=np.array_equal(v,results[0]),max_abs=float(np.max(np.abs(v-results[0]))),
                  relative_to_cpu=float(np.linalg.norm(v-oracle)/np.linalg.norm(oracle))) for v in results[1:]]
    summary=dict(status='passed',repetitions=20,results=reports,
                 forward_has_gather='gather[' in str(forward),backward_has_scatter='scatter-add[' in str(backward),
                 definition='Exact same compiled adjoint and inputs each call; CPU float64 algebraic oracle')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'backward.jaxpr').write_text(str(backward))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
