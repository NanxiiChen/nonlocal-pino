"""Frozen field diagnostic: exact graph BC substitution, no model training."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from scipy import sparse
from examples.corrosion2d.residual_experiment import boundary_values,build_model,load_reference,nondim

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')

def main():
    if jax.default_backend()!='cpu':raise ValueError('Use CPU')
    out={'scope':'Frozen original/s_bc terminal50 s graph q boundary substitution; no retraining or change to free-node output','runs':{}}
    for seed in (42,7):
        run=ROOT/f'20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed{seed}'
        c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
        dc,cle=nondim(c)[-2:];n=ref['X'].size
        points=np.stack([ref['X'].ravel(),ref['Y'].ravel()],axis=-1)
        zero,one,_=boundary_values(c,points);free=~(zero|one)
        row,col,w=(ref[k] for k in ('graph_row','graph_col','graph_weight'))
        matrix=sparse.diags(np.bincount(row,weights=w,minlength=n))-sparse.csr_matrix((w,(row,col)),shape=(n,n))
        model=build_model(c,jax.random.split(jax.random.PRNGKey(seed))[0])
        model=eqx.tree_deserialise_leaves(run/'model_final.eqx',model)
        @eqx.filter_jit
        def evaluate(m):
            x=jnp.asarray(points,dtype=jnp.float32)
            raw=jax.vmap(m.net,in_axes=(0,None))(x,jnp.array(1.))
            def fields(ts):return jax.vmap(lambda t:jax.vmap(m,in_axes=(0,None))(x,t))(ts)
            _,deriv=jax.jvp(fields,(jnp.ones(1),),(jnp.ones(1),))
            return raw,deriv[0,:,1]
        raw,ct=(np.asarray(a) for a in evaluate(model))
        q=cle*.5*(np.tanh(raw[:,1].astype(np.float64))-1)
        imposed=q.copy();imposed[zero]=-cle;imposed[one]=0.
        phi=ref['phi'][-1].ravel();target=ref['c'][-1].ravel()-dc*(3*phi**2-2*phi**3)-cle
        targetdq=matrix@target;old_dq=matrix@q;new_dq=matrix@imposed
        rel=lambda a:float(np.linalg.norm((a-targetdq)[free])/np.linalg.norm(targetdq[free]))
        rms=lambda a:float(np.sqrt(np.mean(a[free]**2)))
        squared_old=np.sum((old_dq[free]-targetdq[free])**2)
        squared_new=np.sum((new_dq[free]-targetdq[free])**2)
        out['runs'][str(seed)]={'path':str(run),'original_Dq_rel_l2_free':rel(old_dq),
            'exact_bc_Dq_rel_l2_free':rel(new_dq),'squared_Dq_error_ratio_exact_bc_to_original':float(squared_new/squared_old),
            'original_CH_rms_free':rms(ct+nondim(c)[0]*old_dq),
            'exact_bc_CH_rms_free':rms(ct+nondim(c)[0]*new_dq),
            'boundary_substitution_effect_Dq_rms_free':rms(new_dq-old_dq),
            'original_Dq_error_rms_free':rms(old_dq-targetdq)}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
