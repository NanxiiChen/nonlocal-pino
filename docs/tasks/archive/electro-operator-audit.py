"""Direct-path CPU checkpoint, input/reference, field and transport audit."""
import json
from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from examples.electropolish2d.operator_training import _build_network, predict_fields
from examples.electropolish2d.operator_data import load_dataset, condition_config
from examples.corrosion2d.residual_experiment import field_metrics, initial_fields, boundary_values

root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
results={}
for suffix in sys.argv[1:]:
    run=root/('20261028-electro-operator-'+suffix+'-01')
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    data,meta,refs=load_dataset(c)
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c.operator,c.operator.image_shape,jax.random.key(c.train.seed)))
    with np.load(run/'prediction.npz') as f:
        saved={k:f[k] for k in f.files}
    np.testing.assert_array_equal(saved['condition_indices'],meta['eval_indices'])
    x=jnp.asarray(data['points'],dtype=jnp.float32);ts=jnp.asarray(refs[0]['time'],dtype=jnp.float32)
    predict=eqx.filter_jit(lambda m,image,known:predict_fields(m,image,x,ts,known,c))
    condition_records=[];max_gap=0.
    for k,(index,ref) in enumerate(zip(meta['eval_indices'],refs)):
        # Independent known initial fields from recorded input geometry identity.
        local=condition_config(c,data['surface_ids'][index],meta['reference_paths'][k])
        initial=initial_fields(local,data['points']);zero,one,bc=boundary_values(c,data['points'])
        initial[zero|one]=bc[zero|one]
        np.testing.assert_allclose(initial,data['initial'][index],rtol=0,atol=3e-7)
        pred=np.asarray(predict(net,jnp.asarray(data['images'][index]),jnp.asarray(data['initial'][index]))).reshape(saved['prediction'][k].shape)
        gap=float(np.max(np.abs(pred-saved['prediction'][k])));max_gap=max(max_gap,gap)
        if jax.default_backend() == 'gpu':
            np.testing.assert_allclose(pred,saved['prediction'][k],rtol=0,atol=2e-5)
        print('field_gap',suffix,index,gap,flush=True)
        target=np.stack([ref['phi'],ref['c']],-1)
        np.testing.assert_array_equal(saved['target'][k],target)
        m=field_metrics(pred,ref)
        frozen=np.broadcast_to(initial.reshape(c.sample.Ny,c.sample.Nx,2),target.shape)
        baseline=field_metrics(frozen,ref)
        pp=pred[...,0];cc=pred[...,1];rp=target[...,0];rc=target[...,1]
        dc=c.physical.c_se-c.physical.c_le;cle=c.physical.c_le
        q=cc-dc*(3*pp**2-2*pp**3)-cle;rq=rc-dc*(3*rp**2-2*rp**3)-cle
        row,col,w=ref['graph_row'],ref['graph_col'],ref['graph_weight'].astype(float)
        n=len(x);A=sparse.diags(np.bincount(row,weights=w,minlength=n))-sparse.csr_matrix((w,(row,col)),shape=(n,n))
        free=~(zero|one);dq=A@q[-1].ravel();rdq=A@rq[-1].ravel()
        m.update(index=index,split='train' if index<c.operator.train_conditions else 'test',
            frozen_initial=baseline,
            terminal_q_relative_l2=float(np.linalg.norm(q[-1]-rq[-1])/np.linalg.norm(rq[-1])),
            terminal_Dq_free_relative_l2=float(np.linalg.norm((dq-rdq)[free])/np.linalg.norm(rdq[free])),
            q_min=float(q.min()),q_max=float(q.max()),reference_q_max=float(rq.max()),
            concentration_min=float(cc.min()),concentration_max=float(cc.max()),
            initial_phi_relative_l2=float(np.linalg.norm(pp[0]-rp[0])/np.linalg.norm(rp[0])),
            boundary_max_abs=float(max(np.abs(pred.reshape(len(ts),n,2)[:,zero]).max(),
                                       np.abs(pred.reshape(len(ts),n,2)[:,one]-1).max())))
        condition_records.append(m)
    selected=[m for m in condition_records if m['split']=='test']
    summary={key:float(np.mean([m[key] for m in selected])) for key in
             ('pooled_rel_l2_phi','terminal_rel_l2_phi','pooled_rel_l2_c','terminal_rel_l2_c',
              'terminal_q_relative_l2','terminal_Dq_free_relative_l2','initial_phi_relative_l2')}
    for key in ('pooled_rel_l2_phi','terminal_rel_l2_phi','pooled_rel_l2_c','terminal_rel_l2_c'):
        logged=json.loads((run/'summary.json').read_text())['final']['test_'+key]
        # CPU FP32 changes must be small relative to the measured prediction error.
        np.testing.assert_allclose(summary[key],logged,rtol=(1e-3 if jax.default_backend() == 'cpu' else 2e-5),atol=1e-7)
    summary['frozen_initial']={key:float(np.mean([m['frozen_initial'][key] for m in selected]))
                              for key in ('pooled_rel_l2_phi','terminal_rel_l2_phi','pooled_rel_l2_c','terminal_rel_l2_c')}
    results[suffix]=dict(run=str(run),test=summary,cpu_gpu_max_field_gap=max_gap,
                        conditions=condition_records,backend=jax.default_backend(),
                        strict_pointwise_2e_5_pass=max_gap<=2e-5,
                        max_reference_time_refinement=max(json.loads(str(r['metadata_json']))['refinement_max_relative_l2'] for r in refs))
    print(suffix,json.dumps(summary), 'replaygap',max_gap,flush=True)
Path('docs/tasks/archive/electro-operator-audit-'+jax.default_backend()+'.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
