"""Two-fixed-input replay, matched initialization and conditioning-path audit."""
import argparse
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.electropolish2d.operator_training import _build_network,predict_fields
from examples.electropolish2d.operator_data import load_dataset
from examples.corrosion2d.residual_experiment import boundary_values

parser=argparse.ArgumentParser()
parser.add_argument('--local-input',choices=('initial','zero'),required=True)
mode=parser.parse_args().local_input
out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
run=root/('20261028-electro-operator-pair-adam-01' if mode=='initial' else
          '20261028-electro-operator-pair-global-adam-01')
c=ConfigDict(json.loads((run/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
data,meta,refs=load_dataset(c)
template=_build_network(c.arch,c.operator.image_shape,jax.random.key(0))
init=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
counts={}
for suffix in ('single-cross-01','cross-01'):
    control=root/('20261028-electro-operator-'+suffix)
    old=eqx.tree_deserialise_leaves(control/'model_initial.eqx',template)
    count=0
    for a,b in zip(jax.tree.leaves(init),jax.tree.leaves(old)):
        if eqx.is_array(a):np.testing.assert_array_equal(a,b);count+=1
    counts[suffix]=count
used=np.load(run/'sampled_conditions.npy')
np.testing.assert_array_equal(np.sort(used,axis=1),np.tile([0,1],(2000,1)))
comparison=root/('20261028-electro-operator-single-cross-01' if mode=='initial' else
                 '20261028-electro-operator-pair-adam-01')
single=json.loads((comparison/'config.json').read_text())
current=json.loads(c.to_json())
# Explicit legacy-schema interpretation for comparison only; disk files intact.
for config in (single,current):
    if 'local_input' not in config['arch']:config['arch']['local_input']='initial'


def diff(a,b,prefix=''):
    assert type(a)==type(b),(prefix,type(a),type(b))
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return sum([diff(a[k],b[k],prefix+'.'+k) for k in a],[])
    return [] if a==b else [prefix.lstrip('.')]


differences=diff(single,current)
expected={'log.run_name','operator.condition_batch','operator.training_indices'} if mode=='initial' else {'log.run_name','arch.local_input'}
assert set(differences)==expected,differences
if mode=='zero':
    np.testing.assert_array_equal(used,np.load(comparison/'sampled_conditions.npy'))
with np.load(run/'prediction.npz') as f:saved={k:f[k] for k in f.files}
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
x=jnp.asarray(data['points'],dtype=jnp.float32);ts=jnp.asarray(refs[0]['time'],dtype=jnp.float32)
fn=eqx.filter_jit(lambda m,image,initial:predict_fields(m,image,x,ts,initial,c))
zero,one,_=boundary_values(c,data['points']);free=~(zero|one)
records=[];replayed={}


def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


for k,(index,ref) in enumerate(zip(meta['eval_indices'],refs)):
    p=np.asarray(fn(net,jnp.asarray(data['images'][index]),jnp.asarray(data['initial'][index])))
    expected=saved['prediction'][k].reshape(p.shape)
    np.testing.assert_allclose(p,expected,rtol=0,atol=2e-5)
    target=np.stack([ref['phi'],ref['c']],-1).reshape(p.shape)
    np.testing.assert_array_equal(target,saved['target'][k].reshape(p.shape))
    row=dict(index=index,replay_gap=float(np.max(abs(p-expected))))
    for channel,name in enumerate(('phi','c')):
        row['pooled_'+name]=rel(p[...,channel],target[...,channel])
        row['terminal_'+name]=rel(p[-1,:,channel],target[-1,:,channel])
    dc=c.physical.c_se-c.physical.c_le;cle=c.physical.c_le
    phase=p[...,0].astype(float);rp=target[...,0]
    q=p[...,1].astype(float)-dc*(3*phase**2-2*phase**3)-cle
    rq=target[...,1]-dc*(3*rp**2-2*rp**3)-cle
    n=len(x);w=ref['graph_weight'].astype(float);i,j=ref['graph_row'],ref['graph_col']
    A=sparse.diags(np.bincount(i,weights=w,minlength=n))-sparse.csr_matrix((w,(i,j)),shape=(n,n))
    dq=A@q[-1];rdq=A@rq[-1]
    row['terminal_q']=rel(q[-1],rq[-1]);row['terminal_Dq']=rel(dq[free],rdq[free])
    records.append(row);replayed[index]=p

# Isolate the two input pathways on the two fitted conditions.
aa,bb=replayed[0],replayed[1]
ba=np.asarray(fn(net,jnp.asarray(data['images'][1]),jnp.asarray(data['initial'][0])))
ab=np.asarray(fn(net,jnp.asarray(data['images'][0]),jnp.asarray(data['initial'][1])))
ra=np.stack([refs[0]['phi'],refs[0]['c']],-1).reshape(aa.shape)
rb=np.stack([refs[1]['phi'],refs[1]['c']],-1).reshape(aa.shape)
swaps=[]
for k,t in enumerate(ts):
    denominator=np.linalg.norm(ra[k,:,0]-rb[k,:,0])
    swaps.append(dict(time=float(t),image=float(.5*(np.linalg.norm(ba[k,:,0]-aa[k,:,0])+
        np.linalg.norm(bb[k,:,0]-ab[k,:,0]))/denominator),
        local=float(.5*(np.linalg.norm(ab[k,:,0]-aa[k,:,0])+
        np.linalg.norm(bb[k,:,0]-ba[k,:,0]))/denominator),
        joint=float(np.linalg.norm(bb[k,:,0]-aa[k,:,0])/denominator)))
summary=json.loads((run/'summary.json').read_text())
for kind in ('pooled','terminal'):
    for field in ('phi','c'):
        np.testing.assert_allclose(np.mean([r[kind+'_'+field] for r in records[:2]]),
            summary['final']['train_'+kind+'_rel_l2_'+field],rtol=2e-5,atol=1e-7)
result=dict(run=str(run),comparison_run=str(comparison),local_input=mode,
    matched_initial_arrays=counts,config_differences=differences,
    actual_condition_indices=np.unique(used).tolist(),updates=len(used),conditions=records,swaps=swaps,
    scope='Two-input fitting, not operator generalization; B/exposure/objective differ from1/16input controls.')
filename='electro-training-pair-audit.json' if mode=='initial' else 'electro-training-global-audit.json'
(out/filename).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('PASS initial arrays',counts,'actual IDs0/1 every update, config and field replay',flush=True)
print('same-input0 phi',records[0]['pooled_phi'],records[0]['terminal_phi'],
      'pair swap t0/.25',swaps[0],swaps[5],flush=True)
