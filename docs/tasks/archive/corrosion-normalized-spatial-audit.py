"""Locate remaining transport error and compare fixed core-pool coverage."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ml_collections import ConfigDict
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from examples.corrosion2d.residual_experiment import boundary_values,build_model,load_reference
from nonlocal_pinn.operators import build_D

RUN=Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed42')

def main():
    if jax.default_backend()!='cpu':raise ValueError('Use CPU')
    c=ConfigDict(json.loads((RUN/'config.json').read_text()));ref=load_reference(c)
    cle=c.physical.c_le;dc=1-cle
    points=np.stack([ref['X'].ravel(),ref['Y'].ravel()],axis=-1);n=len(points)
    z,o,_=boundary_values(c,points);free=~(z|o)
    row,col,w=(ref[k] for k in ('graph_row','graph_col','graph_weight'))
    matrix=sparse.diags(np.bincount(row,weights=w,minlength=n))-sparse.csr_matrix((w,(row,col)),shape=(n,n))
    model=build_model(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
    model=eqx.tree_deserialise_leaves(RUN/'model_final.eqx',model)
    raw=np.asarray(eqx.filter_jit(lambda m:jax.vmap(m.net,in_axes=(0,None))(jnp.asarray(points,dtype=jnp.float32),jnp.array(1.)))(model))
    q=cle*.5*(1+np.tanh(raw[:,1].astype(np.float64)))-cle
    p=ref['phi'][-1].ravel();targetq=ref['c'][-1].ravel()-dc*(3*p*p-2*p*p*p)-cle
    dqerr=matrix@(q-targetq)
    refdq=matrix@targetq
    op=build_D(jnp.asarray(points,dtype=jnp.float32),c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
    np.testing.assert_array_equal(op._row_idx,row);np.testing.assert_array_equal(-np.asarray(op._vals),w)
    coverage=np.zeros(n,dtype=np.int64)
    for i in range(c.sample.K_pool):
        ids,_,mask,_=op.sub_sample(i,c.sample.K_core)
        coverage[np.asarray(ids)[np.asarray(mask)]]+=1
    near=cKDTree(points[z|o]).query(points)[0] <= c.nl.delta+1e-7
    interface=(p>.05)&(p<.95)
    regions={'never_core':free&(coverage==0),'ever_core':free&(coverage>0),'near_dirichlet':free&near,
             'away_dirichlet':free&~near,'interface':free&interface,'bulk':free&~interface}
    total=np.sum(dqerr[free]**2);entries={}
    for name,mask in regions.items():
        entries[name]={'nodes':int(mask.sum()),'Dq_squared_error_fraction':float(np.sum(dqerr[mask]**2)/total),
                       'Dq_error_rms':float(np.sqrt(np.mean(dqerr[mask]**2))) if mask.any() else None,
                       'reference_Dq_rms':float(np.sqrt(np.mean(refdq[mask]**2))) if mask.any() else None,
                       'Dq_relative_l2':float(np.linalg.norm(dqerr[mask])/np.linalg.norm(refdq[mask])) if mask.any() and np.linalg.norm(refdq[mask])>0 else None}
    vals=abs(dqerr[free]);top=max(1,int(np.ceil(.01*len(vals))))
    indices=np.flatnonzero(free);worst=indices[np.argsort(abs(dqerr[free]))[-10:][::-1]]
    record={'run':str(RUN),'scope':'Terminal50 s stable raw-q graph error; fixed training core pool reconstructed exactly',
            'coverage_count_range_free':[int(coverage[free].min()),int(coverage[free].max())],
            'regions':entries,'top_1_percent_squared_Dq_error_fraction':float(np.sum(np.sort(vals**2)[-top:])/total),
            'absolute_Dq_error_quantiles':dict(zip(('p50','p90','p99','max'),np.quantile(vals,[.5,.9,.99,1]).tolist())),
            'worst_nodes':[{'xy':points[i].tolist(),'Dq_error':float(dqerr[i]),'core_occurrences_per_cycle':int(coverage[i]),'reference_phi':float(p[i]),'raw_concentration_output':float(raw[i,1]),'predicted_s':float((q[i]+cle)/cle),'reference_s':float((targetq[i]+cle)/cle),'float32_ds_draw':float(.5*(1-jnp.tanh(jnp.asarray(raw[i,1]))**2))} for i in worst]}
    path=Path(__file__);path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,data,title in zip(axes,(abs(q-targetq)/cle,abs(dqerr),coverage),('q error / c_le','Absolute Dq error','Core occurrences per100 batches')):
        data=np.asarray(data).reshape(ref['X'].shape)
        kwargs={'norm':LogNorm(vmin=max(float(np.max(data))*1e-5,1e-10),vmax=float(np.max(data)))} if ax is not axes[-1] else {}
        im=ax.pcolormesh(ref['X'],ref['Y'],np.maximum(data,1e-12),shading='nearest',**kwargs)
        ax.set(title=title,xlabel='x*',ylabel='y*',aspect='equal');fig.colorbar(im,ax=ax)
    fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
