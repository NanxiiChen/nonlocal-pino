"""Frozen per-geometry PDE diagnostics; reference errors are reporting only.

Raw residual means T*(phi_t+M*A*mu); scaled means P times this same vector.
Midpoint quadrature estimates the original .2-initial/.8-positive PDE objective.
"""
import argparse
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_metrics import metric_job
from examples.confined_ch2d.operator_problem import evaluation_indices,fixed_initial,specification
from examples.confined_ch2d.operator_training import build_network,prepare,phase,apply_operator,make_loss
from examples.confined_ch2d.problem import relative
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled,initial_operator_fields


def diagnose(run,output):
    jax.config.update('jax_enable_x64',True)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if c.operator.metric!='linear_stable_resolvent':
        raise ValueError('This diagnostic explicitly requires the stable-part metric')
    root=Path(c.operator.dataset_path);manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='passed' and manifest['specification']==specification(c)
    monitor,ids=evaluation_indices(manifest);data=[];targets=[]
    for index in ids:
        with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as f:
            d={k:f[k] for k in ('points','volumes','neighbors','coefficients','condition','condition_weights')}
            targets.append(f['phi'])
        d['initial']=fixed_initial(d['points'],c,xp=np)
        if initial_rhs_enabled(c):d['domain_fields']=initial_operator_fields(d,c)
        p=c.physical
        d['metric'],_=metric_job((d['volumes'],d['neighbors'],d['coefficients'],d['initial'],
            p.beta,p.epsilon,p.mobility,p.terminal_time))
        data.append(d)
    nmax=max(len(d['points']) for d in data);kmax=max(d['neighbors'].shape[1] for d in data)
    padded=[]
    for d in data:
        n=len(d['points']);k=d['neighbors'].shape[1]
        item={key:d[key] for key in ('condition','condition_weights')}
        item['points']=np.pad(d['points'],((0,nmax-n),(0,0)))
        if initial_rhs_enabled(c):item['domain_fields']=np.pad(d['domain_fields'],((0,nmax-n),(0,0)))
        for key in ('initial','volumes'):item[key]=np.pad(d[key],(0,nmax-n))
        item['metric']=np.pad(d['metric'],((0,nmax-n),(0,nmax-n)))
        nb=np.repeat(np.arange(nmax)[:,None],kmax,axis=1);nb[:n,:k]=d['neighbors']
        co=np.zeros((nmax,kmax));co[:n,:k]=d['coefficients']
        item['neighbors']=nb;item['coefficients']=co
        padded.append(jax.tree.map(jnp.asarray,item))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,jax.random.key(0)))
    @eqx.filter_jit
    def residual_norms(model,d,times):
        prepared=prepare(model,d)
        def at(t):
            phi,dt=jax.jvp(lambda tt:phase(model,prepared,tt,d,c),(t,),(jnp.ones_like(t),))
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,d)
            raw=c.physical.terminal_time*(dt+c.physical.mobility*apply_operator(mu,d))
            scaled=d['metric']@raw
            return jnp.stack((jnp.dot(d['volumes'],raw**2),jnp.dot(d['volumes'],scaled**2)))/jnp.sum(d['volumes'])
        return jax.vmap(at)(times)
    nt=128;T=c.physical.terminal_time
    times=np.r_[0.,T*(np.arange(nt)+.5)/nt];rows=[]
    for index,d,target in zip(ids,padded,targets,strict=True):
        values=np.asarray(residual_norms(net,d,jnp.asarray(times)))
        fraction=c.loss.initial_residual_fraction if 'initial_residual_fraction' in c.loss else 1/(c.sample.Nt+1)
        loss=fraction*values[0,1]+(1-fraction)*values[1:,1].mean()
        if not rows:
            production=eqx.filter_jit(make_loss(c))(net,jax.tree.map(lambda a:a[None],d),jnp.asarray(times))
            np.testing.assert_allclose(loss,production,rtol=2e-12,atol=1e-13)
        with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
            error=relative(f['phi'],target,np.asarray(d['volumes'])[:target.shape[1]])
        rows.append(dict(index=index,split='train_monitors' if index in monitor else 'development',
            phase_relative_L2=error,expected_objective_midpoint=float(loss),
            initial_preconditioned_squared_RMS=float(values[0,1]),
            positive_preconditioned_squared_RMS=float(values[1:,1].mean()),
            early_preconditioned_squared_RMS=float(values[(times>0)&(times<=.5),1].mean()),
            late_preconditioned_squared_RMS=float(values[times>=T/2,1].mean()),
            positive_raw_squared_RMS=float(values[1:,0].mean())))
    correlations={}
    for split in ('train_monitors','development'):
        chosen=[r for r in rows if r['split']==split]
        x=np.array([r['expected_objective_midpoint'] for r in chosen]);y=np.array([r['phase_relative_L2'] for r in chosen])
        correlations[split]=dict(conditions=len(chosen),spearman=float(spearmanr(x,y).statistic),
            log_Pearson=float(np.corrcoef(np.log(x),np.log(y))[0,1]))
    result=dict(run=str(run),positive_midpoint_nodes=nt,optimizer_updates=0,
        scope='Frozen checkpoint,16training monitors/8reused development domains. No future fields in PDE inputs or gradients. Correlation is diagnostic, not a causal or generalization claim.',
        production_loss_parity=True,correlations=correlations,conditions=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();diagnose(args.run,args.output)
