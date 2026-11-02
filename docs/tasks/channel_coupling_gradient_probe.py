"""Read-only phase/flow loss-gradient geometry of a saved channel checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.losses import make_loss_fn,part_names
from examples.channel_nsch2d.network import build_network


def probe(run,output,time_samples,probe_seed):
    run,output=Path(run),Path(output)
    if not run.is_absolute()or not output.is_absolute():raise ValueError('Absolute source/output paths required')
    if output.exists():raise FileExistsError(output)
    if time_samples<8 or time_samples%8 or probe_seed<0:raise ValueError('Require a positive multiple of8times and nonnegative explicitseed')
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if ('time_derivative'not in c or c.train.compute_dtype!='float32'
            or any(k in c for k in ('balance','stagger','stress_formulation'))):
        raise ValueError('Require an explicit unbalanced original-field FP32 JVP model')
    if tuple(part_names(c))!=('phase','chemical','momentum_x','momentum_y','wall_phase','wall_mu','outlet_traction','outlet_phase','outlet_mu'):
        raise ValueError('Unexpected phase/flow loss ordering')
    summary=json.loads((run/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('Completed saved checkpoint required')
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    net=jax.tree.map(lambda a:a.astype(jnp.float64)if eqx.is_inexact_array(a)else a,net)
    raw=[]
    with np.load(run/'evaluation_inputs.npz',allow_pickle=False)as f:raw.append({k:f[k]for k in f.files})
    with np.load(run/'known_inputs_final.npz',allow_pickle=False)as f:raw.append({k:f[k][0]for k in f.files})
    if set(raw[0])!=set(raw[1]):raise ValueError('Aligned actual input schemas required')
    batch={k:jnp.asarray(np.stack([d[k]for d in raw]),dtype=raw[0][k].dtype
        if np.issubdtype(raw[0][k].dtype,np.integer)else jnp.float64)for k in raw[0]}
    c.train.compute_dtype='float64'
    rng=np.random.default_rng(probe_seed)
    times=(np.arange(time_samples)+rng.random(time_samples))*c.physical.terminal_time/time_samples
    loss=make_loss_fn(c);phase=jnp.asarray([0,1,4,5,7,8]);flow=jnp.asarray([2,3,6])

    @eqx.filter_jit
    def gradients(model,data,ts):
        def objective(n,indices):
            total,parts=loss(n,data,ts)
            return jnp.sum(parts[indices]),(total,parts)
        a,gp=eqx.filter_value_and_grad(lambda n:objective(n,phase),has_aux=True)(model)
        b,gf=eqx.filter_value_and_grad(lambda n:objective(n,flow),has_aux=True)(model)
        return a,b,gp,gf

    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/Path(__file__).name)
    print('Computing full-grid phase/flow gradients with explicit numerical JIT inputs',flush=True)
    accumulated=None;block_metrics=[]
    for start in range(0,time_samples,8):
        a,b,p,f=jax.device_get(gradients(net,batch,jnp.asarray(times[start:start+8])))
        pv,(tv,parts_a)=a;fv,(other_total,parts_b)=b
        np.testing.assert_allclose(parts_a,parts_b,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(tv,other_total,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(pv+fv,tv,rtol=2e-12,atol=2e-12)
        lp=jax.tree.leaves(p);lf=jax.tree.leaves(f)
        p2=sum(np.sum(x*x)for x in lp);f2=sum(np.sum(x*x)for x in lf);dot=sum(np.sum(x*y)for x,y in zip(lp,lf,strict=True))
        block_metrics.append(dict(time_range=[float(times[start]),float(times[start+7])],phase_norm=float(np.sqrt(p2)),flow_norm=float(np.sqrt(f2)),cosine=float(dot/np.sqrt(p2*f2))))
        actual=(pv,fv,tv,parts_a,p,f)
        accumulated=actual if accumulated is None else jax.tree.map(lambda x,y:x+y,accumulated,actual)
        print(json.dumps(dict(completed_times=start+8,total_times=time_samples,cosine=block_metrics[-1]['cosine'])),flush=True)
    phase_value,flow_value,total,parts,gp,gf=jax.tree.map(lambda x:x/(time_samples//8),accumulated)
    left=jax.tree_util.tree_flatten_with_path(gp)[0];right=jax.tree_util.tree_flatten_with_path(gf)[0]
    rows=[];sums={k:np.zeros(3,dtype=np.float64)for k in ('all','shared_trunk','output_head')}
    for (pa,x),(pb,y)in zip(left,right,strict=True):
        if pa!=pb:raise ValueError('Gradient parameter paths disagree')
        if not eqx.is_array(x):continue
        x,y=np.asarray(x),np.asarray(y)
        if x.shape!=y.shape or not np.isfinite(x).all()or not np.isfinite(y).all():raise ValueError('Nonfinite/misaligned actual gradients')
        path=jax.tree_util.keystr(pa);category='output_head'if path.startswith('.output.')else 'shared_trunk'
        stats=np.asarray([np.sum(x*x),np.sum(y*y),np.sum(x*y)])
        sums['all']+=stats;sums[category]+=stats
        rows.append(dict(path=path,shape=list(x.shape),phase_norm=float(np.sqrt(stats[0])),flow_norm=float(np.sqrt(stats[1])),dot=float(stats[2])))
    result={}
    for name,(p2,f2,dot)in sums.items():
        if p2<=0 or f2<=0:raise ValueError('Degenerate gradient group')
        result[name]=dict(phase_norm=float(np.sqrt(p2)),flow_norm=float(np.sqrt(f2)),cosine=float(dot/np.sqrt(p2*f2)),
            phase_gradient_dot_total=float(p2+dot),flow_gradient_dot_total=float(f2+dot))
    report=dict(status='saved_original_JVP_coupling_gradient_probe_completed',source_run=str(run),times=np.asarray(times).tolist(),
        time_samples=time_samples,probe_seed=probe_seed,time_protocol='Independent continuous equalstrata over the fullhorizon; hostFP64 mean of8time blocks at identical parameters',blocks=block_metrics,
        actual_input_labels=['nominal','final_pool_slot0'],domain_nodes_per_input=len(raw[0]['points']),
        part_names=list(part_names(c)),parts=np.asarray(parts).tolist(),phase_group_value=float(phase_value),flow_group_value=float(flow_value),
        groups=result,parameters=rows,supervised_labels=0,optimizer_updates=0,
        precision='Exact saved FP32 model/operator values promoted toCPUFP64; unweighted canonical9parts without causal discount',
        scope='Frozen fullhorizon diagnostic at explicitstratifiedtimes/twoactualgeometryinputs; not historicalSOAP update reconstruction, unique causality, or proof of a successful block-training remedy')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--time-samples',required=True,type=int);p.add_argument('--probe-seed',required=True,type=int)
    a=p.parse_args();probe(a.run,a.output,a.time_samples,a.probe_seed)
