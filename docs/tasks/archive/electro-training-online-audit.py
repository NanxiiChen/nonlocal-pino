"""Matched online image-only validation: stream, initialization, fields and Dq."""
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
from examples.electropolish2d.operator_training import _build_network,predict_fields
from examples.electropolish2d.operator_data import load_dataset
from examples.corrosion2d.residual_experiment import boundary_values

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
base=root/'20261028-electro-operator-online-b8-adam-01'
run=root/'20261028-electro-operator-online-global-adam-01'
cfg=json.loads((run/'config.json').read_text());original=json.loads((base/'config.json').read_text())
assert 'local_input' not in original['arch']
original['arch']['local_input']='initial'  # explicit legacy-schema interpretation


def differences(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        return sum([differences(a[k],b[k],path+'.'+k) for k in a],[])
    return [] if a==b else [path.lstrip('.')]


diff=differences(original,cfg)
assert set(diff)=={'arch.local_input','log.run_name'},diff
c=ConfigDict(cfg)
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
template=_build_network(c.arch,c.operator.image_shape,jax.random.key(0))
init=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
old=eqx.tree_deserialise_leaves(base/'model_initial.eqx',template)
count=0
for a,b in zip(jax.tree.leaves(init),jax.tree.leaves(old)):
    if eqx.is_array(a):np.testing.assert_array_equal(a,b);count+=1
coeff=np.load(run/'online_coefficients.npy');np.testing.assert_array_equal(coeff,np.load(base/'online_coefficients.npy'))
np.testing.assert_array_equal(np.load(run/'online_accepted.npy'),np.load(base/'online_accepted.npy'))
assert coeff.shape==(5000,8,6,3) and len(np.unique(coeff.reshape(40000,18),axis=0))==40000
source={}
for name in ('src/nonlocal_pinn/nn/conditional.py','src/nonlocal_pinn/operators/__init__.py',
             'src/nonlocal_pinn/training/optimizers.py','examples/corrosion2d/residual_experiment.py',
             'examples/corrosion2d/residuals.py'):
    source[name]=(run/'source'/name).read_bytes()==(base/'source'/name).read_bytes()
assert all(source.values())


class RemoveDocstrings(ast.NodeTransformer):
    def generic_visit(self,node):
        super().generic_visit(node)
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
        return node


name='examples/electropolish2d/online_conditions.py'
sampler_ast_equal=ast.dump(RemoveDocstrings().visit(ast.parse((run/'source'/name).read_text())))==ast.dump(RemoveDocstrings().visit(ast.parse((base/'source'/name).read_text())))
assert sampler_ast_equal
data,meta,refs=load_dataset(c)
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
with np.load(run/'prediction.npz') as f:saved={k:f[k] for k in f.files}
x=jnp.asarray(data['points'],dtype=jnp.float32);ts=jnp.asarray(refs[0]['time'],dtype=jnp.float32)
fn=eqx.filter_jit(lambda model,image,initial:predict_fields(model,image,x,ts,initial,c))
zero,one,_=boundary_values(c,data['points']);free=~(zero|one)
records=[]


def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def height(p):
    crossing=(p[:,:-1]>.5)!=(p[:,1:]>.5)
    assert np.all(crossing.sum(1)==1)
    k=crossing.argmax(1);it=np.arange(len(p))[:,None];ix=np.arange(p.shape[2])[None,:]
    lo,hi=p[it,k,ix],p[it,k+1,ix];y=saved['Y'][:,0]
    return y[k]+(.5-lo)/(hi-lo)*(y[k+1]-y[k])


for k,(index,ref) in enumerate(zip(meta['eval_indices'],refs)):
    p=np.asarray(fn(net,jnp.asarray(data['images'][index]),jnp.asarray(data['initial'][index]))).reshape(saved['prediction'][k].shape)
    np.testing.assert_allclose(p,saved['prediction'][k],rtol=0,atol=2e-5)
    target=np.stack([ref['phi'],ref['c']],-1);np.testing.assert_array_equal(target,saved['target'][k])
    entry=dict(index=index,replay_gap=float(np.max(abs(p-saved['prediction'][k]))))
    for channel,label in enumerate(('phi','c')):
        entry['pooled_'+label]=rel(p[...,channel],target[...,channel])
        entry['terminal_'+label]=rel(p[-1,...,channel],target[-1,...,channel])
        entry['initial_'+label]=rel(p[0,...,channel],target[0,...,channel])
    phi=p[...,0].astype(float);rp=target[...,0];dc=c.physical.c_se-c.physical.c_le;cle=c.physical.c_le
    q=p[...,1].astype(float)-dc*(3*phi**2-2*phi**3)-cle
    rq=target[...,1]-dc*(3*rp**2-2*rp**3)-cle
    n=len(x);w=ref['graph_weight'].astype(float);i,j=ref['graph_row'],ref['graph_col']
    A=sparse.diags(np.bincount(i,weights=w,minlength=n))-sparse.csr_matrix((w,(i,j)),shape=(n,n))
    dq=A@q[-1].ravel();rdq=A@rq[-1].ravel()
    entry['terminal_q']=rel(q[-1],rq[-1]);entry['terminal_Dq']=rel(dq[free],rdq[free])
    entry['roughness_prediction']=height(phi).std(-1).tolist();entry['roughness_reference']=height(rp).std(-1).tolist()
    records.append(entry)
summary=json.loads((run/'summary.json').read_text())
for kind in ('pooled','terminal'):
    for field in ('phi','c'):
        np.testing.assert_allclose(np.mean([e[kind+'_'+field] for e in records if e['index']>=16]),
            summary['final']['test_'+kind+'_rel_l2_'+field],rtol=2e-5,atol=1e-7)
use=saved['condition_indices']>=16
spread=[]
for k,t in enumerate(saved['time']):
    p=saved['prediction'][use,k,...,0];r=saved['target'][use,k,...,0]
    spread.append(dict(time=float(t),phi=float(np.linalg.norm(p-p.mean(0))/np.linalg.norm(r-r.mean(0)))))
result=dict(run=str(run),baseline=str(base),config_differences=diff,initial_arrays_equal=count,
    identical_actual_profiles=40000,source_byte_parity=source,sampler_ast_equal_except_docstrings=sampler_ast_equal,
    conditions=records,spread=spread,summary=summary,
    scope='One same-stream matched online control; no causal or stratified sampling; four fixed heldouts, no stability claim.')
(out/'electro-training-online-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('PASS config/init/40000inputs/source and sixGPUfield replays',flush=True)
print('test phi',summary['final']['test_pooled_rel_l2_phi'],summary['final']['test_terminal_rel_l2_phi'],flush=True)
