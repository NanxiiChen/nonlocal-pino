"""Independent CPU reload, transport checks, model gap and full L2 figures."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np
from scipy import ndimage, sparse

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.local_ad import spatial_laplacians
from examples.corrosion2d.local_reference import load, negative_laplacian
from examples.corrosion2d.residuals import _raw_fields_and_derivatives
from examples.corrosion2d.training_precision import cast_floating

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/local-ad-full-training-20261026'
LABELS={'graph_deterministic':'Nonlocal, deterministic',
        'graph_ordinary':'Nonlocal, ordinary',
        'local_tensor':'Local AD, tensor/JVP',
        'local_joint':'Local AD, joint/grad'}


def rel(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def config_differences(a,b,prefix=''):
    differences=[]
    if set(a)!=set(b):raise ValueError('Configuration keys differ: '+prefix)
    for key in a:
        path=f'{prefix}.{key}' if prefix else key
        if isinstance(a[key],dict):differences.extend(config_differences(a[key],b[key],path))
        elif a[key]!=b[key]:differences.append(path)
    return differences


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('Audit must use CPU')
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision','highest')
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('All four full runs must finish first')
    first=ConfigDict(json.loads((Path(manifest['runs'][0]['run'])/'config.json').read_text()))
    graph=r.load_reference(first);local,local_meta=load(first)
    points=np.stack([graph['X'].ravel(),graph['Y'].ravel()],-1)
    zero,one,_=r.boundary_values(first,points);free=~(zero|one)
    row,col,weight=(graph[k] for k in ('graph_row','graph_col','graph_weight'))
    weight=weight.astype(np.float64)
    D=sparse.diags(np.bincount(row,weights=weight,minlength=len(points)))-sparse.csr_matrix((weight,(row,col)),shape=(len(points),len(points)))
    chn,_,_,_,dc,cle=r.nondim(first)
    targets={name:{k:ref[k].reshape(21,-1) for k in ('phi','c')} for name,ref in [('graph',graph),('local',local)]}
    for target in targets.values():target['q']=target['c']-dc*(3*target['phi']**2-2*target['phi']**3)-cle
    targets['graph']['Dq']=(D@targets['graph']['q'].T).T
    with np.load(ROOT/'reports/local-ad-reference-20261026/grid401.npz') as data:
        fine=data['fields'][-1]
    qfine=fine[...,1]-dc*(3*fine[...,0]**2-2*fine[...,0]**3)-cle
    targets['local']['Dq_terminal']=(negative_laplacian(401,201)@qfine.ravel()).reshape(201,401)[::4,::4].ravel()
    results={'scope':'One paired seed42, FP32, 5000 updates, 50 physical seconds; independent CPU final reload',
             'timing_scope':'Fresh weights; existing compilation cache reused. Wall times include actual cache-dependent JIT/evaluation/saves; independent reference generation excluded.',
             'local_boundary':'Extra sixth GradNorm no-flux loss; graph uses its existing truncated-domain operator',
             'reference_paths':{'graph':first.experiment.reference_path,'local':first.experiment.local_reference_path},
             'local_refinement':local_meta,'runs':{},'histories':{},
             'reference_gap_graph_vs_local':{k:{'terminal':rel(targets['graph'][k][-1],targets['local'][k][-1]),
                                               'pooled':rel(targets['graph'][k],targets['local'][k])} for k in ('phi','c','q')}}
    initial0=None;fields={}
    x=jnp.asarray(points,dtype=jnp.float32);times=jnp.asarray(graph['time'],dtype=jnp.float32)

    @eqx.filter_jit
    def evaluate(model):return _raw_fields_and_derivatives(model,x,times)

    # Only terminal spatial AD is required for the independent transport audit.
    @eqx.filter_jit
    def local_dq(model,coords):
        return -jax.vmap(lambda xx:spatial_laplacians(model,xx,jnp.array(1.,dtype=x.dtype),cle)[1])(coords)

    for item in manifest['runs']:
        name=item['label'];run=Path(item['run']);config=ConfigDict(json.loads((run/'config.json').read_text()))
        differences=config_differences(first.to_dict(),config.to_dict())
        allowed={'log.run_name','train.gpu_policy','experiment.spatial_operator','loss.gradnorm_bias'}
        if set(differences)-allowed:raise ValueError(f'Uncontrolled configuration differences: {differences}')
        kn,_=jax.random.split(jax.random.PRNGKey(42));template=r.build_model(config,kn)
        initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
        arrays=[np.asarray(a) for a in jax.tree.leaves(eqx.filter(initial,eqx.is_array))]
        if initial0 is None:initial0=arrays
        else:
            for a,b in zip(arrays,initial0,strict=True):np.testing.assert_array_equal(a,b)
        net=eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',template)
        if (run/'ckpt_005000.eqx').read_bytes()!=(run/'model_final.eqx').read_bytes():raise ValueError('Final checkpoint mismatch')
        if any(a.dtype!=jnp.float32 for a in jax.tree.leaves(eqx.filter(net,eqx.is_array)) if jnp.issubdtype(a.dtype,jnp.floating)):
            raise ValueError('Non-FP32 final parameters')
        raw,_,physical,dt=(np.asarray(v,dtype=np.float64) for v in evaluate(net))
        p,c,ct=(v.T for v in (physical[...,0],physical[...,1],dt[...,1]))
        q=.5*cle*(np.tanh(raw[...,1])-1)
        own='local' if name.startswith('local') else 'graph';target=targets[own]
        prediction_roundoff={}
        with np.load(run/'prediction.npz') as saved:
            for array,key in ((p,'phi'),(c,'c')):
                prediction_roundoff[key]=float(np.max(np.abs(array.reshape(saved[key].shape)-saved[key])))
                np.testing.assert_allclose(array.reshape(saved[key].shape),saved[key],atol=1e-5,rtol=0)
            if own=='local':
                with jax.enable_x64(True):
                    frozen=cast_floating(net,np.dtype('float64'))
                    xx,tt=cast_floating((x,times),np.dtype('float64'))
                    @eqx.filter_jit
                    def frozen_fields(model,coords,ts):
                        return jax.vmap(lambda t:jax.vmap(model,in_axes=(0,None))(coords,t))(ts)
                    fields64=np.asarray(frozen_fields(frozen,xx,tt))
                prediction_roundoff['frozen_fp64_vs_saved_fp32_max_abs']={key:float(np.max(np.abs(fields64[...,i].reshape(saved[key].shape)-saved[key]))) for i,key in enumerate(('phi','c'))}
                prediction_roundoff['frozen_fp64_vs_cpu_fp32_max_abs']={key:float(np.max(np.abs(fields64[...,i]-array))) for key,i,array in [('phi',0,p),('c',1,c)]}
        if own=='graph':dq=(D@q[-1])
        else:
            chunks=[]
            for start in range(0,len(x),256):
                a=x[start:start+256];n=len(a)
                chunks.append(np.asarray(local_dq(net,jnp.pad(a,((0,256-n),(0,0)),mode='edge')))[:n])
            dq=np.concatenate(chunks).astype(np.float64)
        target_dq=target['Dq'][-1] if own=='graph' else target['Dq_terminal']
        ch=ct[-1]+chn*dq
        for a in (p,c,q,dq,ch):
            if not np.isfinite(a).all():raise FloatingPointError(name)
        fields[name]={'phi':p,'c':c,'q':q,'Dq_terminal':dq}
        summary=json.loads((run/'summary.json').read_text())
        history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        if [v['step'] for v in history]!=list(range(0,5001,200)):raise ValueError('Missing full L2 history')
        late=[v for v in history if v['step']>=4000]
        metrics={'run':str(run),'reference_model':own,'updates':5000,
                 'cpu_gpu_frozen_prediction_roundoff':prediction_roundoff,
                 'config_differences_from_default':differences,
                 'end_to_end_seconds':item['wall_seconds'],'trainer_wall_seconds':summary['wall_seconds'],
                 'terminal_phi':rel(p[-1],target['phi'][-1]),'terminal_c':rel(c[-1],target['c'][-1]),
                 'pooled_phi':rel(p,target['phi']),'pooled_c':rel(c,target['c']),
                 'terminal_q_free':rel(q[-1,free],target['q'][-1,free]),
                 'terminal_Dq_free':rel(dq[free],target_dq[free]),
                 'terminal_CH_rms_free':float(np.sqrt(np.mean(ch[free]**2))),
                 'per_time_phi':[rel(a,b) for a,b in zip(p,target['phi'],strict=True)],
                 'per_time_c':[rel(a,b) for a,b in zip(c,target['c'],strict=True)],
                 'liquid_components':[int(ndimage.label(v.reshape(51,101)<.5)[1]) for v in p],
                 'cross_reference_terminal':{ref:{key:rel(array[-1],targets[ref][key][-1]) for key,array in [('phi',p),('c',c),('q',q)]} for ref in targets},
                 'late_4000_5000':{key:{'min':float(min(v[f'terminal_rel_l2_{key}'] for v in late)),
                                        'median':float(np.median([v[f'terminal_rel_l2_{key}'] for v in late])),
                                        'max':float(max(v[f'terminal_rel_l2_{key}'] for v in late))} for key in ('phi','c')}}
        if own=='local':
            metrics['neumann_phi_rms']=summary['final_metrics']['neumann_phi_rms']
            metrics['neumann_q_rms_over_cle']=summary['final_metrics']['neumann_q_rms_over_cle']
        for key in ('phi','c'):
            np.testing.assert_allclose(metrics[f'terminal_{key}'],summary['final_metrics'][f'terminal_rel_l2_{key}'],atol=3e-6,rtol=0)
        metrics['first_evaluation_step_below_phi_1pct']=next((v['step'] for v in history if v['terminal_rel_l2_phi']<.01),None)
        results['runs'][name]=metrics;results['histories'][name]=history
        print(name,json.dumps(metrics),flush=True)
    results['paired_initial_parameters_exact']=True
    results['status']='passed'
    (OUT/'audit.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'audited-fields.npz',X=graph['X'],Y=graph['Y'],time=graph['time'],
                        **{f'{name}_{key}':a for name,f in fields.items() for key,a in f.items()},
                        **{f'{name}_reference_{key}':a for name,f in targets.items() for key,a in f.items()})
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.5),layout='constrained')
    for row,key in enumerate(('phi','c')):
        for name,history in results['histories'].items():
            y=[v[f'terminal_rel_l2_{key}'] for v in history]
            for col,xkey in enumerate(('step','elapsed_seconds')):
                xx=[v[xkey]/(60 if col else 1) for v in history]
                axes[row,col].semilogy(xx,y,label=LABELS[name],marker='.',ms=3)
        for col in range(2):
            ax=axes[row,col];ax.axhline(.01,color='grey',lw=.8,ls=':')
            ax.set(xlabel='Elapsed training minutes (includes JIT/evaluation)' if col else 'Training updates',
                   ylabel=f'{key} relative L2 at t*=1 (50 s)')
            ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Full cold starts: each method versus its matched reference; local AD includes no-flux BC\nFP32 / SOAP / seed 42; graph and local PDE differ; existing compiler cache reused',fontsize=11)
    fig.savefig(OUT/'convergence.png',dpi=180);fig.savefig(OUT/'convergence.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,5),layout='constrained')
    panels=[('Graph reference',targets['graph']['phi']),('Local reference',targets['local']['phi'])]+[(LABELS[k],v['phi']) for k,v in fields.items()]
    for ax,(label,a) in zip(axes.ravel(),panels,strict=True):
        ax.pcolormesh(graph['X'],graph['Y'],a[-1].reshape(51,101),vmin=0,vmax=1,cmap='viridis',shading='auto')
        ax.contour(graph['X'],graph['Y'],a[-1].reshape(51,101),levels=[.5],colors='white',linewidths=.7)
        ax.set(title=label,xlabel='x*',ylabel='y*');ax.set_aspect('equal')
    fig.suptitle('Terminal phase field, t*=1 (50 s)');fig.savefig(OUT/'terminal-phi.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
