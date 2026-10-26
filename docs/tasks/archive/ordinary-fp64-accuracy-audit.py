"""Independent CPU audit and four precision/policy convergence controls."""
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
from scipy import ndimage,sparse

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.residuals import _raw_fields_and_derivatives
from examples.corrosion2d.training_precision import cast_floating,audit_floating

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-fp64-accuracy-20261026'
LOGS=Path('/root/autodl-tmp/tf-logs/corrosion2d')
RUNS={'det_fp32':LOGS/'20261026-full-graph_deterministic-seed42',
      'ordinary_fp32':LOGS/'20261026-full-graph_ordinary-seed42',
      'det_fp64':LOGS/'20261026-precision-float64-phase-seed42',
      'ordinary_fp64':LOGS/'20261026-full-graph_ordinary-float64-seed42'}
LABELS={'det_fp32':'Deterministic FP32','ordinary_fp32':'Ordinary FP32',
        'det_fp64':'Deterministic FP64','ordinary_fp64':'Ordinary FP64'}


def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def flatten(data,prefix=''):
    out={}
    for k,v in data.items():
        path=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,path))
        else:out[path]=v
    return out


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU frozen audit required')
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Finish full training first')
    jax.config.update('jax_default_matmul_precision','highest')
    first=ConfigDict(json.loads((RUNS['det_fp32']/'config.json').read_text()))
    baseline=flatten(first.to_dict());reference=r.load_reference(first)
    coords=np.stack([reference['X'].ravel(),reference['Y'].ravel()],-1)
    zero,one,_=r.boundary_values(first,coords);free=~(zero|one)
    chn,_,_,_,dc,cle=r.nondim(first)
    target={k:reference[k].reshape(21,-1) for k in ('phi','c')}
    target['q']=target['c']-dc*(3*target['phi']**2-2*target['phi']**3)-cle
    row,col=(reference[k] for k in ('graph_row','graph_col'));w=reference['graph_weight'].astype(np.float64)
    D=sparse.diags(np.bincount(row,weights=w,minlength=len(coords)))-sparse.csr_matrix((w,(row,col)),shape=(len(coords),len(coords)))
    target['Dq']=(D@target['q'].T).T
    prior=json.loads((ROOT/'docs/tasks/archive/local-ad-full-training-manifest.json').read_text())
    prior64=json.loads((ROOT/'docs/tasks/archive/corrosion-fp64-accuracy-launch.json').read_text())
    end_times={'det_fp32':prior['runs'][0]['wall_seconds'],'ordinary_fp32':prior['runs'][1]['wall_seconds'],
               'det_fp64':prior64['wall_seconds'],'ordinary_fp64':manifest['wall_seconds']}
    results={'scope':'One paired seed42, same graph/reference and common FP32 input values,5000 updates/50s',
             'reference':first.experiment.reference_path,'new_run':'ordinary_fp64','runs':{},'histories':{},
             'timing_scope':'Observed historical/new end-to-end wall time, includes actual cache-dependent JIT/evaluation/saves; different cache histories; excludes independent reference generation.'}
    fields={};initial0=None
    for name,run in RUNS.items():
        dtype=np.dtype('float64' if name.endswith('fp64') else 'float32')
        jax.config.update('jax_enable_x64',dtype==np.dtype('float64'))
        config=ConfigDict(json.loads((run/'config.json').read_text()))
        current=flatten(config.to_dict());differences={}
        for key in sorted(set(baseline)|set(current)):
            a=baseline[key] if key in baseline else '<absent>'
            b=current[key] if key in current else '<absent>'
            if a!=b:differences[key]={'baseline':a,'run':b}
        allowed={'train.compute_dtype','train.gpu_policy','experiment.spatial_operator',
                 'experiment.local_reference_path','experiment.checkpoint_every','log.run_name'}
        if set(differences)-allowed:raise ValueError(f'Unexpected config differences {differences}')
        if r.reference_spec(config)!=r.reference_spec(first):raise ValueError('Physical/reference specification changed')
        with jax.enable_x64(False):
            kn,_=jax.random.split(jax.random.PRNGKey(42));template32=r.build_model(config,kn)
        template=cast_floating(template32,dtype)
        initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
        arrays=[np.asarray(a,dtype=np.float64) for a in jax.tree.leaves(eqx.filter(initial,eqx.is_array))]
        if initial0 is None:initial0=arrays
        else:
            for a,b in zip(initial0,arrays,strict=True):np.testing.assert_array_equal(a,b)
        net=eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',template)
        parameter_dtype=audit_floating(net,dtype)
        if (run/'ckpt_005000.eqx').read_bytes()!=(run/'model_final.eqx').read_bytes():raise ValueError('Final checkpoint mismatch')
        precision=json.loads((run/'precision.json').read_text())
        if precision['compute_dtype']!=str(dtype):raise ValueError('Precision metadata mismatch')
        for key in ('parameters','parameters_final','data','optimizer_initial','optimizer_final'):
            if precision[key]['dtype']!=str(dtype):raise ValueError(f'Unexpected {key} dtype')
        x=jnp.asarray(coords.astype(np.float32),dtype=dtype);times=jnp.asarray(reference['time'].astype(np.float32),dtype=dtype)
        @eqx.filter_jit
        def evaluate(model):return _raw_fields_and_derivatives(model,x,times)
        raw,_,physical,dt=(np.asarray(v,dtype=np.float64) for v in evaluate(net))
        p,c,ct=(v.T for v in (physical[...,0],physical[...,1],dt[...,1]))
        q=.5*cle*(np.tanh(raw[...,1])-1);dq=(D@q.T).T;ch=ct+chn*dq
        for array in (p,c,q,dq,ch):
            if not np.isfinite(array).all():raise FloatingPointError(name)
        atol=3e-6 if dtype==np.dtype('float32') else 2e-11
        with np.load(run/'prediction.npz') as saved:
            roundoff={key:float(np.max(np.abs(array.reshape(saved[key].shape)-saved[key]))) for key,array in [('phi',p),('c',c)]}
            for key,array in [('phi',p),('c',c)]:np.testing.assert_allclose(array.reshape(saved[key].shape),saved[key],atol=atol,rtol=0)
        summary=json.loads((run/'summary.json').read_text())
        if summary['updates']!=5000 or summary['activation_counts']!={'ac_only':0,'ch_only':0,'joint':5000}:raise ValueError('Wrong update schedule')
        history=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
        if [v['step'] for v in history]!=list(range(0,5001,200)):raise ValueError('Incomplete history')
        late=[v for v in history if v['step']>=4000]
        metric={'run':str(run),'config_differences_from_det_fp32':differences,
                'parameter_dtype':parameter_dtype,'cpu_saved_field_max_abs_difference':roundoff,
                'end_to_end_seconds':end_times[name],'trainer_wall_seconds':summary['wall_seconds'],
                'terminal_phi':rel(p[-1],target['phi'][-1]),'terminal_c':rel(c[-1],target['c'][-1]),
                'pooled_phi':rel(p,target['phi']),'pooled_c':rel(c,target['c']),
                'terminal_q_free':rel(q[-1,free],target['q'][-1,free]),
                'terminal_Dq_free':rel(dq[-1,free],target['Dq'][-1,free]),
                'terminal_CH_rms_free':float(np.sqrt(np.mean(ch[-1,free]**2))),
                'per_time_phi':[rel(a,b) for a,b in zip(p,target['phi'],strict=True)],
                'per_time_c':[rel(a,b) for a,b in zip(c,target['c'],strict=True)],
                'liquid_components_by_time':[int(ndimage.label(v.reshape(51,101)<.5)[1]) for v in p],
                'boundary_q_rmse_over_cle':{'pit':float(np.sqrt(np.mean((q[:,zero]+cle)**2))/cle),'solid':float(np.sqrt(np.mean(q[:,one]**2))/cle)},
                'late_4000_5000':{key:{stat:float(fn([v['terminal_rel_l2_'+logged] for v in late])) for stat,fn in [('min',np.min),('median',np.median),('mean',np.mean),('max',np.max)]} for key,logged in [('phi','phi'),('c','c'),('q_all','q'),('Dq_free','Dq_free')]}}
        for key in ('phi','c'):np.testing.assert_allclose(metric['terminal_'+key],summary['final_metrics']['terminal_rel_l2_'+key],atol=atol,rtol=0)
        metric['best_evaluated_terminal_phi']={k:v for k,v in min(history,key=lambda v:v['terminal_rel_l2_phi']).items() if k in ('step','elapsed_seconds','terminal_rel_l2_phi','terminal_rel_l2_c')}
        metric['first_phi_below_1pct_step']=next((v['step'] for v in history if v['terminal_rel_l2_phi']<.01),None)
        results['runs'][name]=metric;results['histories'][name]=history;fields[name]=(p,c,q)
        print(name,json.dumps(metric),flush=True)
    results['initial_parameter_values_exactly_equal_all_four']=True;results['status']='passed'
    (OUT/'audit.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'audited-fields.npz',X=reference['X'],Y=reference['Y'],time=reference['time'],
                        **{name+'_'+key:a for name,v in fields.items() for key,a in zip(('phi','c','q'),v,strict=True)})
    for keys,filename in [(('phi','c'),'convergence'),(('q','Dq_free'),'transport')]:
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for row,key in enumerate(keys):
            for name,h in results['histories'].items():
                for col,xkey in enumerate(('step','elapsed_seconds')):
                    axes[row,col].semilogy([v[xkey]/(60 if col else 1) for v in h],[v['terminal_rel_l2_'+key] for v in h],label=LABELS[name],marker='.',ms=3)
            for col in range(2):
                ax=axes[row,col];ax.set(xlabel='Elapsed training minutes (actual JIT/eval included)' if col else 'Training updates',ylabel=key+' relative L2 at t*=1 (50 s)')
                ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('Corrosion: same graph / inputs / SOAP, precision x GPU policy, seed 42\n5000 updates; observed wall times have different compiler-cache histories',fontsize=11)
        fig.savefig(OUT/(filename+'.png'),dpi=180)
        if filename=='convergence':fig.savefig(OUT/(filename+'.svg'))
        plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,6.5),layout='constrained')
    for i,(name,arrays) in enumerate([('Reference',(target['phi'],target['c'],target['q'])),('Deterministic FP64',fields['det_fp64']),('Ordinary FP64',fields['ordinary_fp64'])]):
        for j,(key,a) in enumerate(zip(('phi','c','q'),arrays,strict=True)):
            ax=axes[i,j];im=ax.pcolormesh(reference['X'],reference['Y'],a[-1].reshape(51,101),shading='auto',vmin=-cle if key=='q' else 0,vmax=0 if key=='q' else 1,cmap='viridis')
            ax.contour(reference['X'],reference['Y'],arrays[0][-1].reshape(51,101),levels=[.5],colors='white',linewidths=.6)
            ax.set(title=f'{name}: {key}',xlabel='x*',ylabel='y*');ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.75)
    fig.savefig(OUT/'terminal-fields.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
