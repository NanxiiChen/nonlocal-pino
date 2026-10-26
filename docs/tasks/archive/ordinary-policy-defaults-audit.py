"""Frozen CPU field audit for explicit ordinary-policy default selection."""
import argparse
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
from examples.corrosion2d.residuals import _raw_fields_and_derivatives
from examples.corrosion2d.training_precision import cast_floating, audit_floating

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-policy-defaults-20261026'
LOGS=Path('/root/autodl-tmp/tf-logs')


def rel(a,b):
    norm=np.linalg.norm(b)
    if norm<=0:raise ValueError('Positive reference norm required')
    return float(np.linalg.norm(a-b)/norm)


def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        path=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,path))
        else:out[path]=v
    return out


def read(run):
    config=ConfigDict(json.loads((run/'config.json').read_text()))
    summary=json.loads((run/'summary.json').read_text())
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    if summary['updates']!=5000 or [v['step'] for v in history]!=list(range(0,5001,200)):
        raise ValueError(f'Incomplete5000-update run: {run}')
    if (run/'ckpt_005000.eqx').read_bytes()!=(run/'model_final.eqx').read_bytes():
        raise ValueError('Final checkpoint differs')
    return config,summary,history


def late(history,keys):
    rows=[v for v in history if v['step']>=4000]
    if len(rows)!=6:raise ValueError('Need all six late evaluations')
    return {key:{name:float(fn([v[key] for v in rows])) for name,fn in
                 [('min',np.min),('median',np.median),('mean',np.mean),('max',np.max)]} for key in keys}


def kks_specs():
    specs=[]
    for case in ('electropolish2d','corrosion2d'):
        for seed in (42,2026):
            if case=='electropolish2d':
                old='20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed42' if seed==42 else '20261025-independent-seed2026-default-strategy'
            else:
                old='20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42' if seed==42 else '20261026-objective-phase-seed2026'
            specs.append((case,'deterministic','float32',seed,LOGS/case/old))
        for dtype in (('float32','float64') if case=='electropolish2d' else ('float64',)):
            for seed in (42,2026):
                name='20261026-full-graph_ordinary-float64-seed42' if case=='corrosion2d' and seed==42 else f'20261026-defaultcheck-ordinary-{dtype}-seed{seed}'
                specs.append((case,'ordinary',dtype,seed,LOGS/case/name))
    return specs


def audit_kks():
    results={'scope':'5000updates; unchanged original residuals, physical parameters, graph/reference, model and optimizer within each case. Seeds42/2026 are run provenance. CPU replay of every final checkpoint.',
             'timing_scope':'Trainer elapsed includes actual cache-dependent compilation, evaluations and saves; cache histories differ. Use isolated benchmark for steady-state speed.',
             'runs':{},'histories':{}}
    fields={};initials={};baselines={}
    for case,policy,dtype_name,seed,run in kks_specs():
        name=f'{case}-{policy}-{dtype_name}-{seed}'
        dtype=np.dtype(dtype_name);jax.config.update('jax_enable_x64',dtype_name=='float64')
        config,summary,history=read(run)
        reference=r.load_reference(config)
        current=flatten(config.to_dict())
        if case not in baselines:baselines[case]=(current,r.reference_spec(config))
        baseline,spec=baselines[case]
        diff={k:{'baseline':baseline[k] if k in baseline else '<absent>', 'run':current[k] if k in current else '<absent>'} for k in set(baseline)|set(current) if (baseline[k] if k in baseline else '<absent>')!=(current[k] if k in current else '<absent>')}
        allowed={'train.seed','train.gpu_policy','train.compute_dtype','experiment.checkpoint_every','experiment.local_reference_path','experiment.spatial_operator','log.run_name'}
        inactive={'experiment.lr_tail_start':0,'experiment.ch_rms_after':0,'experiment.lr_tail_end':1e-5}
        for key,value in inactive.items():
            if key in diff:
                if diff[key]['baseline']!='<absent>' or diff[key]['run']!=value:raise ValueError('Changed continuation')
                allowed.add(key)
        if set(diff)-allowed or r.reference_spec(config)!=spec:raise ValueError(f'Unexpected config/reference changes: {diff}')
        with jax.enable_x64(False):
            template32=r.build_model(config,jax.random.split(jax.random.PRNGKey(seed))[0])
        template=cast_floating(template32,dtype)
        initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
        arrays=[np.asarray(a,dtype=np.float64) for a in jax.tree.leaves(eqx.filter(initial,eqx.is_array))]
        pair=(case,seed)
        if pair in initials:
            for a,b in zip(initials[pair],arrays,strict=True):np.testing.assert_array_equal(a,b)
        else:initials[pair]=arrays
        net=eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',template)
        params=audit_floating(net,dtype)
        precision=None
        if (run/'precision.json').is_file():
            precision=json.loads((run/'precision.json').read_text())
            for key in ('parameters','parameters_final','data','optimizer_initial','optimizer_final'):
                if precision[key]['dtype']!=dtype_name:raise ValueError(f'Wrong dtype: {key}')
        X,Y=reference['X'],reference['Y'];coords=np.stack([X.ravel(),Y.ravel()],-1)
        x=jnp.asarray(coords.astype(np.float32),dtype=dtype);times=jnp.asarray(reference['time'].astype(np.float32),dtype=dtype)
        @eqx.filter_jit
        def evaluate(model):return _raw_fields_and_derivatives(model,x,times)
        raw,_,physical,dt=(np.asarray(a,dtype=np.float64) for a in evaluate(net))
        p,c,ct=(a.T for a in (physical[...,0],physical[...,1],dt[...,1]))
        chn,_,_,_,dc,cle=r.nondim(config)
        target={k:reference[k].reshape(21,-1) for k in ('phi','c')}
        target['q']=target['c']-dc*(3*target['phi']**2-2*target['phi']**3)-cle
        row,col=(reference[k] for k in ('graph_row','graph_col'));w=reference['graph_weight'].astype(np.float64)
        D=sparse.diags(np.bincount(row,weights=w,minlength=len(coords)))-sparse.csr_matrix((w,(row,col)),shape=(len(coords),len(coords)))
        q=.5*cle*(np.tanh(raw[...,1])-1);dq=(D@q.T).T;rdq=(D@target['q'].T).T
        zero,one,_=r.boundary_values(config,coords);free=~(zero|one)
        for a in (p,c,q,dq,ct):
            if not np.isfinite(a).all():raise FloatingPointError(name)
        atol=5e-6 if dtype_name=='float32' else 2e-11
        with np.load(run/'prediction.npz') as saved:
            replay={key:float(np.max(np.abs(a.reshape(saved[key].shape)-saved[key]))) for key,a in [('phi',p),('c',c)]}
            for key,a in [('phi',p),('c',c)]:np.testing.assert_allclose(a.reshape(saved[key].shape),saved[key],atol=atol,rtol=0)
        metric={'run':str(run),'case':case,'policy':policy,'dtype':dtype_name,'seed_as_run_provenance':seed,'config_differences_from_old_seed42':diff,
                'parameter_dtype':params,'precision_metadata':precision,'cpu_saved_max_abs_difference':replay,'trainer_wall_seconds':summary['wall_seconds'],
                'terminal_phi':rel(p[-1],target['phi'][-1]),'terminal_c':rel(c[-1],target['c'][-1]),'pooled_phi':rel(p,target['phi']),'pooled_c':rel(c,target['c']),
                'terminal_q_free':rel(q[-1,free],target['q'][-1,free]),'terminal_Dq_free':rel(dq[-1,free],rdq[-1,free]),
                'terminal_CH_rms_free':float(np.sqrt(np.mean((ct[-1,free]+chn*dq[-1,free])**2))),
                'per_time_phi':[rel(a,b) for a,b in zip(p,target['phi'],strict=True)],'per_time_c':[rel(a,b) for a,b in zip(c,target['c'],strict=True)],
                'late_4000_5000':late(history,['terminal_rel_l2_'+key for key in ('phi','c','q','Dq_free')])}
        for key in ('phi','c'):np.testing.assert_allclose(metric['terminal_'+key],summary['final_metrics']['terminal_rel_l2_'+key],atol=atol,rtol=0)
        shaped=p.reshape(21,*X.shape)
        if case=='electropolish2d':
            counts=((shaped[:,:-1]-.5)*(shaped[:,1:]-.5)<0).sum(axis=1)
            metric['topology']={'single_interface_every_column_every_time':bool(np.all(counts==1)),'vertical_crossings_by_time':[{str(int(k)):int(v) for k,v in zip(*np.unique(a,return_counts=True),strict=True)} for a in counts]}
        else:metric['topology']={'liquid_components_by_time':[int(ndimage.label(a<.5)[1]) for a in shaped]}
        fields[name]={'phi':p,'c':c,'q':q};results['runs'][name]=metric;results['histories'][name]=history
        print(name,metric['terminal_phi'],metric['terminal_c'],metric['topology'],flush=True)
        (OUT/'kks-audit-partial.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    results['paired_initial_values_exactly_equal']=True;results['status']='passed'
    (OUT/'kks-audit.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'kks-audited-fields.npz',**{name+'_'+key:a for name,data in fields.items() for key,a in data.items()})
    for case in ('electropolish2d','corrosion2d'):
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for name,h in results['histories'].items():
            if not name.startswith(case):continue
            label=name.removeprefix(case+'-')
            for row,key in enumerate(('phi','c')):
                for col,xkey in enumerate(('step','elapsed_seconds')):
                    axes[row,col].semilogy([v[xkey]/(60 if col else 1) for v in h],[v['terminal_rel_l2_'+key] for v in h],label=label,lw=1.3)
        for row,key in enumerate(('phi','c')):
            for col in range(2):
                axes[row,col].set(xlabel='Actual elapsed training minutes' if col else 'Updates',ylabel=key+' relative L2 at t*=1')
                axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=7)
        fig.suptitle(case+': all predeclared runs; raw evaluations,5000updates\nElapsed times include different compiler-cache histories')
        fig.savefig(OUT/(case+'-convergence.png'),dpi=170);plt.close(fig)


def audit_melting():
    from examples.icemelting3d.train import _build_phase_network
    case='icemelting3d';run=LOGS/case/'20261026-defaultcheck-ordinary-float32-seed42'
    jax.config.update('jax_enable_x64',False)
    config,summary,history=read(run)
    with np.load(config.eval.graph_reference_path,allow_pickle=False) as archive:ref={k:archive[k] for k in archive.files}
    identity=json.loads(str(ref['identity']));n=config.sample.Nx
    if n!=128 or identity['physical']!=config.physical.to_dict() or identity['nl']!=config.nl.to_dict() or identity['grid']!=[n,n,n]:raise ValueError('Melting reference identity mismatch')
    template=_build_phase_network(config,jax.random.split(jax.random.PRNGKey(config.train.seed))[0])
    model=eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',template)
    axis=np.asarray(jnp.linspace(0.,1.,n));fold=(np.abs(np.arange(n)-(n-1)/2)-.5).astype(int)
    predict=eqx.filter_jit(lambda net,x,t:jax.vmap(net,in_axes=(0,None))(x,t)[:,0])
    prediction=np.empty(n**3,dtype=np.float64);chunk=config.eval.chunk_size
    for start in range(0,n**3,chunk):
        ids=np.arange(start,min(start+chunk,n**3));ijk=np.stack(np.unravel_index(ids,(n,n,n)),axis=1);points=axis[ijk[:,[1,0,2]]];actual=len(points)
        if actual<chunk:points=np.pad(points,((0,chunk-actual),(0,0)),mode='edge')
        prediction[start:start+actual]=np.asarray(predict(model,jnp.asarray(points),jnp.float32(1)))[:actual]
    prediction=prediction.reshape(n,n,n);truth=ref['phi'][-1][np.ix_(fold,fold,fold)].swapaxes(0,1)
    terminal=rel(prediction,truth);np.testing.assert_allclose(terminal,summary['final_metrics']['graph_terminal_full_rel_l2'],atol=2e-7,rtol=.005)
    result={'status':'passed','run':str(run),'seed_as_run_provenance':42,'policy':config.train.gpu_policy,'xla_flags':summary['xla_flags'],
            'terminal_graph_full_phi_relative_l2_cpu':terminal,'terminal_solid_fraction_relative_l2':rel(prediction+1,truth+1),
            'solid_volume_fraction':float(np.mean(prediction>0)),'reference_solid_volume_fraction':float(np.mean(truth>0)),
            'maximum_phase_absolute_error':float(np.max(np.abs(prediction-truth))),'trainer_wall_seconds':summary['training_wall_seconds'],
            'late_sampled_graph_phi':late(history,['graph_terminal_rel_l2']),'history':history}
    (OUT/'melting-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['kks','melting']);args=parser.parse_args()
    if jax.default_backend()!='cpu':raise RuntimeError('Frozen CPU audit only')
    jax.config.update('jax_default_matmul_precision','highest')
    audit_kks() if args.stage=='kks' else audit_melting()
