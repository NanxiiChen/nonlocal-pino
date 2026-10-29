"""Frozen per-condition gradients/head groups and image/local input swaps."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.corrosion2d.residual_experiment import boundary_values,grid,nondim
from examples.electropolish2d.operator_data import load_dataset
from examples.electropolish2d.operator_training import _build_network,make_loss_vector,predict_fields
from examples.electropolish2d.online_conditions import make_sampler

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
run=root/'20261028-electro-operator-online-b8-adam-01'
c=ConfigDict(json.loads((run/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
data,meta,refs=load_dataset(c)
keys=jr.split(jr.PRNGKey(c.train.seed),3)
template=_build_network(c.arch,c.operator.image_shape,keys[0])
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
single=eqx.tree_deserialise_leaves(root/'20261028-electro-operator-single-cross-01/model_final.eqx',template)
xx,yy,points=grid(c);x=jnp.asarray(points,dtype=jnp.float32)
op=build_D(x,c.nl.delta,method='sparse',kernel=c.nl.kernel)
zero,one,_=boundary_values(c,points);free=jnp.asarray(~(zero|one))
ic=np.random.default_rng(c.train.seed).choice(len(points),c.operator.ic_points,replace=False)
bc=np.r_[np.flatnonzero(zero),np.flatnonzero(one)]
ids,sub,core,_=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)[0]
queries=x[jnp.concatenate([ids,jnp.asarray(ic),jnp.asarray(bc)])]
mask=core & free[ids]
sample,geometry=make_sampler(c)
images,known,coeff,accepted=sample(jr.fold_in(keys[2],0),queries,geometry(queries))
assert int(accepted)>=8
np.testing.assert_array_equal(coeff,np.load(run/'online_coefficients.npy')[0])
loss=make_loss_vector(c,nondim(c),len(ic),int(zero.sum()))
labels=['AC','CH','IC'];loss_ids=[0,1,4]

# Partition flattened parameter gradients by declared network field, plus each
# raw output row. All groups are diagnostics, not a train-time freeze.
groups={};offset=0
for path,leaf in jax.tree_util.tree_flatten_with_path(eqx.filter(net,eqx.is_array))[0]:
    if not eqx.is_array(leaf):continue
    name=path[0].name
    groups.setdefault(name,[]).extend(range(offset,offset+leaf.size))
    if name=='output':
        assert leaf.shape[0]==2
        for channel in range(2):
            per=leaf.size//2
            groups.setdefault('output_phi' if channel==0 else 'output_liquid',[]).extend(
                range(offset+channel*per,offset+(channel+1)*per))
    offset+=leaf.size
groups={k:np.array(v,dtype=int) for k,v in groups.items()}


@eqx.filter_jit
def condition_grad(model,times,image,local):
    values,pb=eqx.filter_vjp(lambda m:loss(m,times,(image[None],local[None],queries),sub,mask),model)
    gradients=[]
    for i in loss_ids:
        g=pb(jax.nn.one_hot(i,5))[0]
        gradients.append(jnp.concatenate([a.ravel() for a in jax.tree.leaves(g) if eqx.is_array(a)]))
    return values[jnp.array(loss_ids)],jnp.stack(gradients)


_,kt=jr.split(jr.fold_in(keys[1],0))
original_times=jr.uniform(kt,(c.sample.Nt,),minval=c.sample.t_min,maxval=1.)
result=dict(run=str(run),gradient_probes=[],swaps=[],scope='Frozen-model probes, not optimizer histories or new training data.')


def save():
    (out/'electro-training-conditioning.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


for name,times in [('recorded',original_times),('early',jnp.array([.001,.003,.005,.02]))]:
    grads=[];values=[]
    for image,local in zip(images,known):
        v,g=condition_grad(net,times,image,local)
        grads.append(np.asarray(g,dtype=float));values.append(np.asarray(v,dtype=float))
    g=np.stack(grads)
    assert g.shape==(8,3,offset) and np.isfinite(g).all()
    record=dict(times_kind=name,times=np.asarray(times).tolist(),losses=np.asarray(values).tolist(),equations={})
    for i,label in enumerate(labels):
        individual=g[:,i];norm=np.linalg.norm(individual,axis=-1);mean=individual.mean(0)
        cosine=individual@individual.T/norm[:,None]/norm[None,:]
        upper=cosine[np.triu_indices(8,1)]
        group_norms={key:np.linalg.norm(individual[:,idx],axis=-1).tolist() for key,idx in groups.items()}
        record['equations'][label]=dict(norm_of_mean=float(np.linalg.norm(mean)),
            mean_of_norms=float(norm.mean()),retention=float(np.linalg.norm(mean)/norm.mean()),
            cosine=cosine.tolist(),pair_cosine_mean=float(upper.mean()),
            pair_cosine_min=float(upper.min()),pair_cosine_negative_fraction=float(np.mean(upper<0)),
            individual_gradient_l2=norm.tolist(),group_gradient_l2=group_norms,
            mean_gradient_l1=float(abs(mean).sum()))
    result['gradient_probes'].append(record);save()
    print('gradients',name,{key:dict(retention=v['retention'],cos=v['pair_cosine_mean'])
        for key,v in record['equations'].items()},flush=True)

# Inference swaps isolate image and local-IC pathways. Mixed pairs are
# intentionally inconsistent inputs for sensitivity only, never training.
ts=jnp.asarray(refs[0]['time'],dtype=jnp.float32)
predict=eqx.filter_jit(lambda m,image,local:predict_fields(m,image,x,ts,local,c))
for model_name,model in [('online_final',net),('single_input0_final',single)]:
    for first,second in ((0,1),(16,17)):
        im_a,im_b=jnp.asarray(data['images'][first]),jnp.asarray(data['images'][second])
        loc_a,loc_b=jnp.asarray(data['initial'][first]),jnp.asarray(data['initial'][second])
        aa=np.asarray(predict(model,im_a,loc_a));ba=np.asarray(predict(model,im_b,loc_a))
        ab=np.asarray(predict(model,im_a,loc_b));bb=np.asarray(predict(model,im_b,loc_b))
        ref_a=refs[meta['eval_indices'].index(first)];ref_b=refs[meta['eval_indices'].index(second)]
        actual_a=np.stack([ref_a['phi'],ref_a['c']],-1).reshape(aa.shape)
        actual_b=np.stack([ref_b['phi'],ref_b['c']],-1).reshape(aa.shape)
        values=[]
        for k,t in enumerate(ts):
            row=dict(time=float(t))
            for i,label in enumerate(('phi','c')):
                denominator=np.linalg.norm(actual_a[k,:,i]-actual_b[k,:,i])
                assert denominator>0
                row[label]=dict(image_swap=float(.5*(np.linalg.norm(ba[k,:,i]-aa[k,:,i])+
                    np.linalg.norm(bb[k,:,i]-ab[k,:,i]))/denominator),
                    local_swap=float(.5*(np.linalg.norm(ab[k,:,i]-aa[k,:,i])+
                    np.linalg.norm(bb[k,:,i]-ba[k,:,i]))/denominator),
                    joint_change=float(np.linalg.norm(bb[k,:,i]-aa[k,:,i])/denominator))
            values.append(row)
        result['swaps'].append(dict(model=model_name,interfaces=[first,second],by_time=values));save()
        print('swap',model_name,first,second,'t0',values[0]['phi'],'t.25',values[5]['phi'],flush=True)

# Existing independently evaluated batch8 means check separate-condition AD
# against the original vmap/mean loss probe, with FP32 batching tolerance.
prior=json.loads((out/'electro-online-balance-diagnosis.json').read_text())
late=next(g for g in prior['gradients'] if g['checkpoint']=='final' and g['actual_online_iteration_zero_based']==0)
comparison={}
for i,label in zip(loss_ids,labels):
    value=result['gradient_probes'][0]['equations'][label]
    relative=abs(value['norm_of_mean']/late['gradient_l2'][i]-1)
    assert relative<.01,(label,relative)
    comparison[label]=relative
result['separate_vs_original_batch_gradient_norm_relative_gap']=comparison
save()
print('COMPLETE batching check',comparison,flush=True)
