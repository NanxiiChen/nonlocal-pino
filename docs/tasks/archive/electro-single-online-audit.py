"""Final GPU checkpoint replay and independent same-graph field/stream audit."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from examples.electropolish2d.operator_data import load_dataset
from examples.electropolish2d.operator_training import _build_network,predict_fields
from examples.corrosion2d.residual_experiment import field_metrics,boundary_values
from examples.electropolish2d.losses import _rough_surface

jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision','highest')
root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
records={}
for suffix in ('single-cross-01','single-soap-01','online-b8-adam-01'):
    run=root/('20261028-electro-operator-'+suffix)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    data,metadata,refs=load_dataset(c)
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c.arch,c.operator.image_shape,jax.random.key(0)))
    with np.load(run/'prediction.npz') as f:saved={k:f[k] for k in f.files}
    x=jnp.asarray(data['points'],dtype=jnp.float32);ts=jnp.asarray(refs[0]['time'],dtype=jnp.float32)
    fn=eqx.filter_jit(lambda m,im,initial:predict_fields(m,im,x,ts,initial,c))
    gaps=[];metrics=[]
    for k,(index,ref) in enumerate(zip(metadata['eval_indices'],refs)):
        p=np.asarray(fn(net,jnp.asarray(data['images'][index]),jnp.asarray(data['initial'][index]))).reshape(saved['prediction'][k].shape)
        gaps.append(float(np.max(abs(p-saved['prediction'][k]))))
        np.testing.assert_allclose(p,saved['prediction'][k],rtol=0,atol=2e-5)
        target=np.stack([ref['phi'],ref['c']],-1)
        np.testing.assert_array_equal(target,saved['target'][k])
        m=field_metrics(p,ref);m['condition_index']=index
        phase=p[...,0];conc=p[...,1];rp=target[...,0];rc=target[...,1]
        dc=c.physical.c_se-c.physical.c_le;cle=c.physical.c_le
        q=conc-dc*(3*phase**2-2*phase**3)-cle;rq=rc-dc*(3*rp**2-2*rp**3)-cle
        row,col,w=ref['graph_row'],ref['graph_col'],ref['graph_weight'].astype(float)
        n=len(x);A=sparse.diags(np.bincount(row,weights=w,minlength=n))-sparse.csr_matrix((w,(row,col)),shape=(n,n))
        zero,one,_=boundary_values(c,data['points']);free=~(zero|one)
        dq=A@q[-1].ravel();rdq=A@rq[-1].ravel()
        crossings=np.sum((phase[:,:-1]<.5)!=(phase[:,1:]<.5),axis=1)
        m.update(phase_min=float(phase.min()),phase_max=float(phase.max()),
                 c_min=float(conc.min()),c_max=float(conc.max()),
                 q_terminal_relative_l2=float(np.linalg.norm(q[-1]-rq[-1])/np.linalg.norm(rq[-1])),
                 Dq_terminal_free_relative_l2=float(np.linalg.norm((dq-rdq)[free])/np.linalg.norm(rdq[free])),
                 minimum_vertical_crossings=int(crossings.min()),maximum_vertical_crossings=int(crossings.max()))
        metrics.append(m)
    info=dict(run=str(run),gpu_max_field_gap=max(gaps),conditions=metrics,
        parameter_array_values=sum(a.size for a in jax.tree.leaves(net) if eqx.is_array(a)),
        frozen_frequency_values=net.spatial_frequencies.size)
    if 'online' in c.operator:
        coeff=np.load(run/'online_coefficients.npy');accepted=np.load(run/'online_accepted.npy')
        assert coeff.shape==(5000,8,6,3);assert accepted.shape==(5000,)
        assert np.isfinite(coeff).all() and accepted.min()>=8
        unique=len(np.unique(coeff.reshape(-1,18),axis=0));assert unique==40000
        assert not (run/'sampled_conditions.npy').exists()
        selected=[metrics[metadata['eval_indices'].index(i)] for i in metadata['test_indices']]
        sampled_heights=[]
        for iteration in (0,24,1234,4999):
            for case in range(8):
                profile=_rough_surface(np.linspace(0,1,2000),c.surface.y0,coeff[iteration,case].astype(float))
                assert profile.min()>c.operator.online.height_min-1e-6
                assert profile.max()<c.operator.online.height_max+1e-6
                sampled_heights.append([float(profile.min()),float(profile.max())])
        info['stream']=dict(shape=list(coeff.shape),unique_profiles=unique,
            accepted_candidate_min=int(accepted.min()),accepted_candidate_max=int(accepted.max()),
            independent_numpy_height_checks=len(sampled_heights),sampled_height_extrema=sampled_heights)
        info['memory']=json.loads((run/'memory.json').read_text())
    else:
        used=np.load(run/'sampled_conditions.npy');np.testing.assert_array_equal(used,0)
        info['actual_training_ids']=np.unique(used).tolist()
        selected=[metrics[0]]
    summary=json.loads((run/'summary.json').read_text())
    for key in ('pooled_rel_l2_phi','terminal_rel_l2_phi','pooled_rel_l2_c','terminal_rel_l2_c'):
        actual=float(np.mean([m[key] for m in selected]))
        logged=summary['final'][('test_' if 'online' in c.operator else 'train_')+key]
        np.testing.assert_allclose(actual,logged,rtol=2e-5,atol=1e-7)
    records[suffix]=info
    print(suffix,'max replay gap',max(gaps),'phi',np.mean([m['pooled_rel_l2_phi'] for m in selected]),np.mean([m['terminal_rel_l2_phi'] for m in selected]),flush=True)
Path('docs/tasks/archive/electro-single-online-audit.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
