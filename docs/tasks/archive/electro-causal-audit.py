"""Independent matched-input, causal recurrence, field/transport replay audit."""
import ast
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.electropolish2d.operator_training import _build_network, predict_fields
from examples.electropolish2d.operator_data import load_dataset
from examples.corrosion2d.residual_experiment import boundary_values

out=Path('docs/tasks/archive')
root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
base=root/'20261028-electro-operator-online-b8-adam-01'
run=root/'20261028-electro-operator-online-b8-causal-01'
cfg=json.loads((run/'config.json').read_text()); old=json.loads((base/'config.json').read_text())
c=ConfigDict(cfg)
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)


def differences(a,b,prefix=''):
    result=[]
    if isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys()
        for key in a:result.extend(differences(a[key],b[key],prefix+'.'+key))
    elif a!=b:result.append(prefix.lstrip('.'))
    return result


diff=differences(old,cfg)
assert set(diff)=={'loss.use_causal','log.run_name'},diff
template=_build_network(c.arch,c.operator.image_shape,jax.random.key(0))
init=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
oldinit=eqx.tree_deserialise_leaves(base/'model_initial.eqx',template)
init_count=0
for a,b in zip(jax.tree.leaves(init),jax.tree.leaves(oldinit)):
    if eqx.is_array(a):np.testing.assert_array_equal(a,b);init_count+=1
coeff=np.load(run/'online_coefficients.npy'); oldcoeff=np.load(base/'online_coefficients.npy')
np.testing.assert_array_equal(coeff,oldcoeff)
assert coeff.shape==(5000,8,6,3)
assert len(np.unique(coeff.reshape(40000,18),axis=0))==40000
np.testing.assert_array_equal(np.load(run/'online_accepted.npy'),np.load(base/'online_accepted.npy'))
source={}
for name in ('src/nonlocal_pinn/nn/conditional.py','src/nonlocal_pinn/operators/__init__.py',
    'src/nonlocal_pinn/training/optimizers.py','examples/corrosion2d/residual_experiment.py',
    'examples/corrosion2d/residuals.py','examples/electropolish2d/online_conditions.py',
    'examples/electropolish2d/configs/train_operator_online.py'):
    source[name]=(run/'source'/name).read_bytes()==(base/'source'/name).read_bytes()
config_source='examples/electropolish2d/configs/train_operator_online.py'
assert all(equal for name,equal in source.items() if name!=config_source)
# The baseline source predates an explanatory reference-bank-only comment.
config_ast_equal=(ast.dump(ast.parse((run/'source'/config_source).read_text()))==
                  ast.dump(ast.parse((base/'source'/config_source).read_text())))
assert config_ast_equal
assert (run/'source/src/nonlocal_pinn/training/causal.py').read_bytes()==Path('src/nonlocal_pinn/training/causal.py').read_bytes()
def function_ast(path,name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
assert function_ast(run/'source/examples/electropolish2d/operator_training.py','physical_fields')==function_ast(base/'source/examples/electropolish2d/operator_training.py','physical_fields')

logs=[json.loads(l) for l in (run/'loss.jsonl').read_text().splitlines()]
keys=jax.random.split(jax.random.PRNGKey(c.train.seed),3)
for e in logs:
    a=e['causal'];eps=np.min(a['eps_used'])
    segments=np.stack([a['segments_ac'],a['segments_ch']],axis=1)
    actual=np.stack([a['weights_ac'],a['weights_ch']],axis=1)
    expected=np.exp(-eps*np.concatenate([np.zeros((*segments.shape[:2],1)),np.cumsum(segments[:,:,:-1],axis=-1)],axis=-1))
    np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=2e-7)
    np.testing.assert_allclose(np.mean(actual*segments,axis=(0,2)),e['parts'][:2],rtol=5e-5,atol=2e-7)
    np.testing.assert_allclose(segments.mean(axis=(0,2)),e['unweighted_parts'][:2],rtol=5e-5,atol=2e-7)
    np.testing.assert_array_equal(actual[:,:,0],1)
    assert np.all(np.diff(actual,axis=-1)<=1e-7)
    eps_next=np.array(a['eps_used'])
    for i,w in enumerate(actual.mean(0)):
        if w[-1]>c.causal.max_last_weight and eps_next[i]<c.causal.max_eps:eps_next[i]*=c.causal.step_size
        if w.mean()<c.causal.min_mean_weight:eps_next[i]/=c.causal.step_size
    np.testing.assert_allclose(a['eps_next'],np.clip(eps_next,0,c.causal.max_eps),rtol=2e-6,atol=1e-7)
    _,kt=jax.random.split(jax.random.fold_in(keys[1],e['step']-1))
    ts=jax.random.uniform(kt,(c.sample.Nt,),minval=c.sample.t_min,maxval=1.)
    np.testing.assert_array_equal(a['sorted_times'],np.sort(np.asarray(ts)))

data,meta,refs=load_dataset(c)
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
with np.load(run/'prediction.npz') as f:saved={k:f[k] for k in f.files}
x=jnp.asarray(data['points'],dtype=jnp.float32);ts=jnp.asarray(refs[0]['time'],dtype=jnp.float32)
fn=eqx.filter_jit(lambda m,im,known:predict_fields(m,im,x,ts,known,c))
zero,one,_=boundary_values(c,data['points']);free=~(zero|one)
records=[]


def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def height(phase):
    crossing=(phase[:,:-1]>.5)!=(phase[:,1:]>.5)
    assert np.all(crossing.sum(1)==1)
    k=crossing.argmax(1);it=np.arange(phase.shape[0])[:,None];ix=np.arange(phase.shape[2])[None,:]
    lo,hi=phase[it,k,ix],phase[it,k+1,ix]
    y=saved['Y'][:,0]
    return y[k]+(.5-lo)/(hi-lo)*(y[k+1]-y[k])


for j,(index,ref) in enumerate(zip(meta['eval_indices'],refs)):
    p=np.asarray(fn(net,jnp.asarray(data['images'][index]),jnp.asarray(data['initial'][index]))).reshape(saved['prediction'][j].shape)
    np.testing.assert_allclose(p,saved['prediction'][j],rtol=0,atol=2e-5)
    target=np.stack([ref['phi'],ref['c']],-1);np.testing.assert_array_equal(target,saved['target'][j])
    info=dict(index=index,replay_gap=float(np.max(abs(p-saved['prediction'][j]))))
    for field,name in enumerate(('phi','c')):
        info['pooled_'+name]=rel(p[...,field],target[...,field])
        info['terminal_'+name]=rel(p[-1,...,field],target[-1,...,field])
        info['initial_'+name]=rel(p[0,...,field],target[0,...,field])
    phase=p[...,0].astype(float);rp=target[...,0]
    dc=c.physical.c_se-c.physical.c_le;cle=c.physical.c_le
    q=p[...,1].astype(float)-dc*(3*phase**2-2*phase**3)-cle
    rq=target[...,1]-dc*(3*rp**2-2*rp**3)-cle
    n=len(x);w=ref['graph_weight'].astype(float);row,col=ref['graph_row'],ref['graph_col']
    A=sparse.diags(np.bincount(row,weights=w,minlength=n))-sparse.csr_matrix((w,(row,col)),shape=(n,n))
    dq=(A@q.reshape(len(ts),n).T).T;rdq=(A@rq.reshape(len(ts),n).T).T
    info['terminal_q']=rel(q[-1],rq[-1]);info['terminal_Dq']=rel(dq[-1,free],rdq[-1,free])
    hp,hr=height(phase),height(rp)
    info['roughness_prediction']=hp.std(-1).tolist();info['roughness_reference']=hr.std(-1).tolist()
    info['height_prediction']=hp.mean(-1).tolist();info['height_reference']=hr.mean(-1).tolist()
    info['c_range']=[float(p[...,1].min()),float(p[...,1].max())]
    records.append(info)
test=[v for v in records if v['index']>=16]
summary=json.loads((run/'summary.json').read_text())
for kind in ('pooled','terminal'):
    for field in ('phi','c'):
        np.testing.assert_allclose(np.mean([v[kind+'_'+field] for v in test]),summary['final']['test_'+kind+'_rel_l2_'+field],rtol=2e-5,atol=1e-7)
spreads=[]
indices=np.flatnonzero(saved['condition_indices']>=16)
for k,t in enumerate(saved['time']):
    row=dict(time=float(t))
    for i,field in enumerate(('phi','c')):
        p=saved['prediction'][indices,k,...,i];r=saved['target'][indices,k,...,i]
        row[field]=float(np.linalg.norm(p-p.mean(0))/np.linalg.norm(r-r.mean(0)))
    spreads.append(row)
result=dict(run=str(run),baseline=str(base),config_differences=diff,
    identical_initial_parameter_arrays=init_count,identical_actual_profiles=40000,
    source_byte_parity=source,online_config_ast_equal=config_ast_equal,
    causal_records_validated=len(logs),
    conditions=records,spread=spreads,summary=summary,
    scope='One matched cold-start pair, final fixed budget; no test checkpoint selection or stability claim.')
(out/'electro-causal-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('PASS configs, init arrays',init_count,'40000exact inputs, causal records',len(logs),'fields and errors',flush=True)
print('final',summary['final'],flush=True)
