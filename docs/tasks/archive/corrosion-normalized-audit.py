"""Independent CPU replay of normalized-data controls and transport fidelity."""
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
from scipy import sparse
from examples.corrosion2d.residual_experiment import boundary_values, build_model, load_reference, nondim, training_boundary_data
from examples.corrosion2d.residuals import compute_pde
from examples.corrosion2d.train import _bc_info

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
RUNS={f'{form}/{mode}/seed42':ROOT/f'20261025-data-{mode}-icmatched_graph-bcmatched_graph-{form}-stagger0-seed42'
      for form in ('phase','latent') for mode in ('physical','s_bc','s_bc_ic')}
RUNS.update({f'phase/{mode}/seed7':ROOT/f'20261025-data-{mode}-icmatched_graph-bcmatched_graph-phase-stagger0-seed7'
             for mode in ('physical','s_bc')})

def h(p):return 3*p*p-2*p*p*p

def rms(a):return float(np.sqrt(np.mean(a*a)))

def rel(a,b):
    den=np.linalg.norm(b)
    if den==0:raise ValueError('Undefined relative norm')
    return float(np.linalg.norm(a-b)/den)

def flat(d,prefix=''):
    result={}
    for k,v in d.items():
        key=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):result.update(flat(v,key))
        else:result[key]=v
    return result

def main():
    if jax.default_backend()!='cpu':raise ValueError('Set JAX_PLATFORMS=cpu')
    baseline=RUNS['phase/physical/seed42']
    basecfg=flat(json.loads((baseline/'config.json').read_text()))
    snapshots=json.loads((baseline/'source/files.json').read_text())
    sources={p:(baseline/'source'/p).read_bytes() for p in snapshots}
    initial_by_seed={seed:(RUNS[f'phase/physical/seed{seed}']/'model_initial.eqx').read_bytes() for seed in (42,7)}
    out={'scope':'Six seed42 controls plus paired seed7 validation; independent CPU raw/checkpoint/graph audit','runs':{}}
    histories={};predictions={}
    for label,run in RUNS.items():
        c=ConfigDict(json.loads((run/'config.json').read_text()))
        cfg=flat(c.to_dict())
        allowed={'log.run_name','experiment.formulation','experiment.data_loss_mode','train.seed'}
        differences=[k for k in basecfg.keys()|cfg.keys() if k not in allowed and (k not in basecfg or k not in cfg or basecfg[k]!=cfg[k])]
        if differences:raise ValueError(f'Unmatched configs {label}:{differences}')
        assert (run/'model_initial.eqx').read_bytes()==initial_by_seed[c.train.seed]
        for path,source in sources.items():assert (run/'source'/path).read_bytes()==source,(label,path)
        summary=json.loads((run/'summary.json').read_text())
        assert summary['updates']==5000 and summary['initial_training_nodes']==5151
        assert summary['xla_flags']=='--xla_gpu_deterministic_ops=true'
        assert summary['initial_sampling']=='matched_graph' and summary['boundary_sampling']=='matched_graph'
        assert summary['activation_counts']=={'ac_only':0,'ch_only':0,'joint':5000}
        ref=load_reference(c);nd=nondim(c);dc,cle=nd[-2:]
        n=ref['X'].size
        targetp,targetc=ref['phi'].reshape(21,n),ref['c'].reshape(21,n)
        targetq=targetc-dc*h(targetp)-cle;targetcl=targetq+cle
        with np.load(run/'prediction.npz') as saved:
            np.testing.assert_array_equal(saved['time'],ref['time'])
            p,conc=saved['phi'].astype(np.float64).reshape(21,n),saved['c'].astype(np.float64).reshape(21,n)
        fields={}
        for name,a,b in (('phi',p,targetp),('c',conc,targetc)):
            fields['terminal_rel_l2_'+name]=rel(a[-1],b[-1])
            fields['pooled_rel_l2_'+name]=rel(a,b)
            for key in ('terminal_rel_l2_'+name,'pooled_rel_l2_'+name):
                np.testing.assert_allclose(fields[key],summary['final_metrics'][key],rtol=1e-12)
        points=np.stack([ref['X'].ravel(),ref['Y'].ravel()],axis=-1)
        zero,one,_=boundary_values(c,points);free=~(zero|one)
        row,col,weight=(ref[k] for k in ('graph_row','graph_col','graph_weight'))
        diag=np.bincount(row,weights=weight,minlength=n)
        matrix=sparse.diags(diag)-sparse.csr_matrix((weight,(row,col)),shape=(n,n))
        x=jnp.asarray(points,dtype=jnp.float32);ts=jnp.asarray(ref['time'],dtype=jnp.float32)
        hold_pit,hold_solid=_bc_info(c.pits.entries,c.sample.N_bc_dir,c.sample.N_bc_solid)
        pit,solid=training_boundary_data(c,points,hold_pit,hold_solid)
        xb=jnp.concatenate([pit,solid,hold_pit,hold_solid])
        key=jax.random.split(jax.random.PRNGKey(c.train.seed))[0]
        net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_model(c,key))
        @eqx.filter_jit
        def evaluate(m):
            raw=jax.vmap(lambda t:jax.vmap(m.net,in_axes=(0,None))(x,t))(ts)
            br=jax.vmap(lambda t:jax.vmap(m.net,in_axes=(0,None))(xb,t))(ts)
            def at_time(times):
                return jax.vmap(lambda t:jax.vmap(m,in_axes=(0,None))(x,t))(times)
            _,deriv=jax.jvp(at_time,(jnp.ones(1),),(jnp.ones(1),))
            sub=(jnp.asarray(row),jnp.asarray(col),-jnp.asarray(weight),jnp.asarray(diag,dtype=jnp.float32))
            _,native_ch=compute_pde(m,x,jnp.ones(1),sub,nd,'phase',.5)
            return raw,m.constraint(raw),br,m.constraint(br),deriv[0,:,1],native_ch[:,0]
        raw,replay,br,bfields,ct,native_ch=(np.asarray(v) for v in evaluate(net))
        np.testing.assert_allclose(replay[...,0],p,atol=5e-6,rtol=0)
        np.testing.assert_allclose(replay[...,1],conc,atol=5e-6,rtol=0)
        cl=cle*.5*(1+np.tanh(raw[...,1].astype(np.float64)));q=cl-cle
        bcq=cle*.5*(1+np.tanh(br[...,1].astype(np.float64)))-cle
        dq=(matrix@q.T).T;targetdq=(matrix@targetq.T).T
        # Exact decomposition of the saved physical concentration errors.
        reconstructed=conc-dc*h(p)
        np.testing.assert_allclose(conc-targetc,dc*(h(p)-h(targetp))+reconstructed-targetcl,atol=1e-15,rtol=0)
        transport={
            'terminal_cL_rel_l2':rel(cl[-1],targetcl[-1]),'pooled_cL_rel_l2':rel(cl,targetcl),
            'terminal_q_rel_l2':rel(q[-1],targetq[-1]),'pooled_q_rel_l2':rel(q,targetq),
            'terminal_q_rel_l2_free':rel(q[-1,free],targetq[-1,free]),
            'terminal_Dq_rel_l2_free':rel(dq[-1,free],targetdq[-1,free]),
            'pooled_Dq_rel_l2_free':rel(dq[:,free],targetdq[:,free]),
            'initial_Dq_rel_l2_free':rel(dq[0,free],targetdq[0,free]),
            'initial_q_rmse_over_cle_free':rms((q[0]-targetq[0])[free])/cle,
            'reference_terminal_ct_rms_free':rms(-nd[0]*targetdq[-1,free]),
            'terminal_ct_rel_l2_free':rel(ct[free],-nd[0]*targetdq[-1,free]),
            'stable_full_CH_rms_free':rms((ct+nd[0]*dq[-1])[free]),
            'native_full_CH_rms_free':rms(native_ch[free]),
            'reference_terminal_mean_c_t_free':float(np.mean(-nd[0]*targetdq[-1,free])),
            'predicted_terminal_mean_c_t_free':float(np.mean(ct[free])),
            'stable_terminal_mean_CH_residual_free':float(np.mean((ct+nd[0]*dq[-1])[free])),
            'terminal_zero_cL_derivative_fraction':float(np.mean(np.asarray(jnp.tanh(jnp.asarray(raw[-1,:,1])))**2==1)),
            'terminal_cL_over_cle_range':[float(cl[-1].min()/cle),float(cl[-1].max()/cle)],
        }
        bcs={}
        start_hold=len(pit)+len(solid)
        for name,sl,target in (('pit',slice(0,len(pit)),0.),('solid',slice(len(pit),start_hold),1.),
                               ('held_out_pit',slice(start_hold,start_hold+len(hold_pit)),0.),
                               ('held_out_solid',slice(start_hold+len(hold_pit),None),1.)):
            bcs[name]={'physical_phi_max_error':float(np.max(abs(bfields[:,sl,0]-target))),
                       'physical_c_max_error':float(np.max(abs(bfields[:,sl,1]-target))),
                       'q_rmse_over_cle':rms((bcq[:,sl]-(target-1)*cle)/cle),
                       'q_max_error_over_cle':float(np.max(abs(bcq[:,sl]-(target-1)*cle))/cle)}
        arc=(targetp[0]>.05)&(targetp[0]<.95)&free
        ic={'phi_rmse_free':rms((p[0]-targetp[0])[free]),'c_rmse_free':rms((conc[0]-targetc[0])[free]),
            'q_rmse_over_cle_arc':rms((q[0]-targetq[0])[arc])/cle,
            'Dq_rel_l2_arc':rel(dq[0,arc],targetdq[0,arc])}
        history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        late=[r for r in history if r['step']>=4000]
        log=[json.loads(line) for line in (run/'loss_history.jsonl').read_text().splitlines()]
        for r in log:
            assert all(r['active']) and all(v>0 and np.isfinite(v) for v in r['adaptive_weights'])
            np.testing.assert_array_equal(r['weights'],r['adaptive_weights'])
        result={'path':str(run),'configuration_initialization_sources_verified':True,'data_loss_mode':c.experiment.data_loss_mode,
                'physical_metrics':fields,'transport':transport,'boundary_sample_metrics':bcs,'initial_metrics':ic,
                'late_terminal_phi_median':float(np.median([v['terminal_rel_l2_phi'] for v in late])),
                'late_terminal_c_median':float(np.median([v['terminal_rel_l2_c'] for v in late])),
                'late_terminal_q_median':float(np.median([v['terminal_rel_l2_q'] for v in late])),
                'late_terminal_Dq_median':float(np.median([v['terminal_rel_l2_Dq_free'] for v in late])),
                'late_terminal_Dq_range':[float(min(v['terminal_rel_l2_Dq_free'] for v in late)),float(max(v['terminal_rel_l2_Dq_free'] for v in late))],
                'cpu_field_replay_max_abs_difference':float(np.max(abs(replay-np.stack([p,conc],axis=-1)))),
                'BC_scope':f'{len(pit)} pit/{len(solid)} solid training graph nodes and held-out random128/128 BC points; maximum/RMSE over21 times',
                'transport_scope':'CPU raw network cL in float64; graph action independent SciPy; native CH and c_t are FP32 AD'}
        out['runs'][label]=result;histories[label]=history;predictions[label]=p.reshape(ref['phi'].shape)
        print(label,fields,transport['terminal_q_rel_l2_free'],transport['terminal_Dq_rel_l2_free'],flush=True)
    path=Path(__file__);path.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,key,title in zip(axes.ravel(),('phi','c','q','Dq_free'),('phase','total concentration','concentration drive q','transport Dq')):
        for label,history in histories.items():
            ax.semilogy([r['step'] for r in history],[r['terminal_rel_l2_'+key] for r in history],label=label)
        ax.set(xlabel='Updates',ylabel='50 s relative L2',title=title);ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
    seed42_predictions={label:p for label,p in predictions.items() if label.endswith('/seed42')}
    assert len(seed42_predictions)==6
    for ax,(label,p) in zip(axes.ravel(),seed42_predictions.items()):
        image=ax.pcolormesh(ref['X'],ref['Y'],p[-1],vmin=0,vmax=1,shading='nearest')
        ax.contour(ref['X'],ref['Y'],ref['phi'][-1],levels=[.5],colors='white',linewidths=.8)
        ax.set(title=label,xlabel='x*',ylabel='y*',aspect='equal')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Phase at50 s; white=reference interface')
    fig.savefig(path.with_name(path.stem+'-fields.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    seed7_predictions={label:p for label,p in predictions.items() if label.endswith('/seed7')}
    assert len(seed7_predictions)==2
    for ax,(label,p) in zip(axes,seed7_predictions.items()):
        image=ax.pcolormesh(ref['X'],ref['Y'],p[-1],vmin=0,vmax=1,shading='nearest')
        ax.contour(ref['X'],ref['Y'],ref['phi'][-1],levels=[.5],colors='white',linewidths=.8)
        ax.set(title=label,xlabel='x*',ylabel='y*',aspect='equal')
    fig.colorbar(image,ax=axes.tolist(),label='Phase at50 s; white=reference interface')
    fig.savefig(path.with_name(path.stem+'-validation-fields.png'),dpi=150);plt.close(fig)

if __name__=='__main__':main()
