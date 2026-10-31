"""Input-only frozen output-layer gradient variance across geometry and time.

Prepared features are cached ONLY because the checkpoint is frozen. No optimizer
update or evolved reference values are used. This is a partial-gradient probe,
not a measurement of every encoder parameter or a proof of an accuracy floor.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,prepare,phase,apply_operator,make_loss
from examples.confined_ch2d.operator_problem import fixed_initial
from examples.confined_ch2d.operator_metrics import linear_stable_resolvent
from nonlocal_pinn.training.runtime import configure_gpu_policy


def probe(run,step,output,counts):
    start=time.perf_counter();c=ConfigDict(json.loads((run/'config.json').read_text()))
    if c.operator.metric!='linear_stable_resolvent':raise ValueError('Probe requires the actual stable-part metric')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    root=Path(c.operator.dataset_path);manifest=json.loads((root/'manifest.json').read_text());ids=list(range(16));raw=[]
    for i in ids:
        with np.load(root/manifest['conditions'][i]['file'],allow_pickle=False) as f:
            d={k:f[k] for k in ('points','volumes','neighbors','coefficients','condition','condition_weights')}
        d['initial']=fixed_initial(d['points'],c,xp=np)
        p=c.physical;d['metric'],_=linear_stable_resolvent(d['volumes'],d['neighbors'],d['coefficients'],d['initial'],p.beta,p.epsilon,p.mobility,p.terminal_time)
        raw.append(d)
    nmax=max(len(d['points']) for d in raw);kmax=max(d['neighbors'].shape[1] for d in raw);data=[]
    for d in raw:
        n,k=d['neighbors'].shape;nb=np.repeat(np.arange(nmax)[:,None],kmax,axis=1);co=np.zeros((nmax,kmax))
        nb[:n,:k]=d['neighbors'];co[:n,:k]=d['coefficients']
        item=dict(points=np.pad(d['points'],((0,nmax-n),(0,0))),volumes=np.pad(d['volumes'],(0,nmax-n)),
            initial=np.pad(d['initial'],(0,nmax-n)),neighbors=nb,coefficients=co,
            metric=np.pad(d['metric'],((0,nmax-n),(0,nmax-n))),condition=d['condition'],condition_weights=d['condition_weights'])
        data.append(jax.tree.map(jnp.asarray,item))
    net=eqx.tree_deserialise_leaves(run/f'ckpt_{step:06d}.eqx',build_network(c,jax.random.key(0)))
    prepare_frozen=eqx.filter_jit(prepare);prepared=[prepare_frozen(net,d) for d in data]
    initial_weight=1/(c.sample.Nt+1)
    def loss_w(w,model,h,d,times):
        model=eqx.tree_at(lambda x:x.output.weight,model,w)
        def at(t):
            phi,dt=jax.jvp(lambda x:phase(model,h,x,d,c),(t,),(jnp.ones_like(t),))
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,d)
            r=d['metric']@(c.physical.terminal_time*(dt+c.physical.mobility*apply_operator(mu,d)))
            return jnp.dot(d['volumes'],r*r)/d['volumes'].sum()
        values=jax.vmap(at)(times)
        return initial_weight*values[0]+(1-initial_weight)*jnp.mean(values[1:])
    evaluate=eqx.filter_jit(jax.value_and_grad(loss_w,argnums=0))
    key=jax.random.split(jax.random.key(c.train.seed),3)[2];rows=[]
    for nt in counts:
        gradients=[];losses=[]
        for i,(d,h) in enumerate(zip(data,prepared,strict=True)):
            sample=[];values=[]
            for j in range(8):
                draws=jax.random.uniform(jax.random.fold_in(key,step+j+1),(nt,),dtype=jnp.float64)
                times=jnp.concatenate((jnp.zeros(1),c.physical.terminal_time*(jnp.arange(nt)+draws)/nt))
                value,gradient=evaluate(net.output.weight,net,h,d,times)
                sample.append(np.asarray(gradient).ravel());values.append(float(value))
                if nt==4 and i==0 and j==0:
                    full=float(make_loss(c)(net,jax.tree.map(lambda a:a[None],d),times))
                    np.testing.assert_allclose(value,full,atol=1e-12,rtol=1e-12)
            gradients.append(sample);losses.append(values)
        g=np.asarray(gradients);geo_mean=g.mean(axis=0);mean=g.mean(axis=(0,1))
        geometry_var=float(np.mean(np.sum((g-geo_mean[None,:,:])**2,axis=-1))*len(ids)/(len(ids)-1))
        shared_time_var=float(np.mean(np.sum((geo_mean-mean)**2,axis=-1))*8/7)
        norms=dict(geometry_single_variance=geometry_var,shared_time_variance=shared_time_var,
            mean_gradient_norm=float(np.linalg.norm(mean)),mean_loss=float(np.mean(losses)))
        norms['estimated_batch4_shared_variance']=geometry_var/4+shared_time_var
        norms['estimated_batch16_shared_variance']=geometry_var/16+shared_time_var
        norms['estimated_batch4_independent_variance']=(geometry_var+shared_time_var)/4
        rows.append(dict(positive_times=nt,**norms));print(json.dumps(rows[-1]),flush=True)
    result=dict(run=str(run),step=step,backend=jax.default_backend(),conditions=ids,time_sets=8,
        initial_pde_weight=initial_weight,diagnostics=rows,wall_seconds=time.perf_counter()-start,
        optimizer_updates=0,future_fields_used=False,
        scope='Only frozen final-layer weight gradients on16traininginputs; finite-cohort variance estimates, not a full-network proof')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--positive-times',type=int,nargs='+',required=True)
    a=p.parse_args();probe(a.run,a.step,a.output,a.positive_times)
