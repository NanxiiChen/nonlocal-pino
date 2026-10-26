"""Actual GPU legacy equivalence and frozen-model transport evaluation probe."""
import importlib.util
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.corrosion2d import residual_experiment as runner
from examples.corrosion2d import residuals
from examples.corrosion2d import train as production
from nonlocal_pinn.operators import build_D

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed42')

def main():
    c=ConfigDict(json.loads((ROOT/'config.json').read_text()))
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    spec=importlib.util.spec_from_file_location('saved_residuals',ROOT/'source/examples/corrosion2d/residuals.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    xx,yy,p=runner.grid(c);x=jnp.asarray(p,dtype=jnp.float32)
    op=build_D(x,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
    ids,sub,mask,_=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)[0]
    zero,one,_=runner.boundary_values(c,p);free=~(zero|one)
    mask=mask & jnp.asarray(free)[ids]
    z,o=production._bc_info(c.pits.entries,c.sample.N_bc_dir,c.sample.N_bc_solid)
    z,o=runner.training_boundary_data(c,p,z,o)
    xi,ic=runner.training_initial_data(c,p,p[:1])
    nd=runner.nondim(c)
    losses=[m.make_normalized_data_loss_vector(c,nd,xi,ic,z,o,normalize_initial=False) for m in (old,residuals)]
    kn,kt=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(ROOT/'model_initial.eqx',runner.build_model(c,kn))
    t=jax.random.uniform(jax.random.fold_in(kt,0),(c.sample.Nt,),minval=c.sample.t_min,maxval=1.)
    class Recorder:
        def update(self,grad,state,params):return jax.tree.map(jnp.zeros_like,grad),grad
    results=[]
    for f in losses:
        step=runner.make_step(f,Recorder(),c)
        result=step(net,None,jnp.ones(5),t,x[ids],sub,mask,jnp.array(True),jnp.ones(5,dtype=bool))
        results.append((np.asarray(result[3]),np.asarray(result[2]),np.concatenate([np.asarray(v).ravel() for v in jax.tree.leaves(result[1]) if eqx.is_array(v)])))
    checks={label:bool(np.array_equal(a,b)) for label,a,b in zip(('losses','weights','gradients'),results[0],results[1])}
    if not all(checks.values()):raise ValueError(f'Legacy first-batch mismatch: {checks}')
    full=(op._row_idx,op._col_idx,op._vals,op._diag)
    final=eqx.tree_deserialise_leaves(ROOT/'model_final.eqx',net)
    @eqx.filter_jit
    def terminal(m):
        raw=jax.vmap(m.net,in_axes=(0,None))(x,jnp.array(1.))
        def at_times(ts):return jax.vmap(lambda tau:jax.vmap(m,in_axes=(0,None))(x,tau))(ts)
        _,deriv=jax.jvp(at_times,(jnp.ones(1),),(jnp.ones(1),))
        values=[residuals.compute_transport_pde(m,x,jnp.ones(1),full,nd,'phase',.5,transport_evaluation=e)[1][:,0] for e in ('physical_difference','kks_raw_edges')]
        return raw,deriv[0,:,1],*values
    raw,ct,legacy,edges=(np.asarray(a) for a in terminal(final))
    q=.5*nd[-1]*(np.tanh(raw[:,1].astype(np.float64))-1)
    stable=ct+nd[0]*(runner.graph_matrix(op,len(x))@q)
    rms=lambda a:float(np.sqrt(np.mean(a[free]**2)))
    output={'source_run':str(ROOT),'backend':jax.default_backend(),
        'legacy_full_batch_exact':checks,
        'terminal_CH_rms':{'legacy':rms(legacy),'raw_edges':rms(edges),'float64_graph_raw_q':rms(stable)},
        'terminal_CH_evaluation_error_rms':{'legacy':rms(legacy-stable),'raw_edges':rms(edges-stable)},
        'scope':'Old/new legacy first-batch losses/weights/parameter gradients; frozen winner evaluation only, not retraining evidence'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
