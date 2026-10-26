"""Final15-model independent CPU field/transport/source/rate audit."""
import json
import time
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
from scipy.spatial import cKDTree
from examples.corrosion2d.residual_experiment import boundary_values, build_model, load_reference, nondim, training_boundary_data
from examples.corrosion2d.residuals import compute_pde, compute_transport_pde
from examples.corrosion2d.train import _bc_info

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
PLAN=[('phase',e,n) for e in ('physical_difference','kks_raw_edges') for n in ('l1','rms')]
PLAN.append(('latent','kks_raw_edges','rms'))
PLAN.append(('bounded','kks_raw_edges','l1'))
RUNS={f'{f}/{e}/{n}/seed42':ROOT/f'20261025-transport-{e}-ch{n}-s_bc-icmatched_graph-bcmatched_graph-{f}-seed42'
      for f,e,n in PLAN}
RUNS.update({f'prior/{f}/s_bc/seed42':ROOT/f'20261025-data-s_bc-icmatched_graph-bcmatched_graph-{f}-stagger0-seed42'
             for f in ('phase','latent')})
RUNS['prior/phase/s_bc/seed7']=ROOT/'20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed7'
# The final audit requires all three explicit completed validation/refinement
# records. Missing or partial records fail rather than omitting controls.
VALIDATION_RECORD=Path(__file__).with_name('corrosion-transport-validation.json')


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
    validation=json.loads(VALIDATION_RECORD.read_text())
    assert len(validation)==3,'Final audit requires all three validation/refinement controls'
    for v in validation:
        if v['returncode']!=0:raise ValueError('Validation failure')
        RUNS[v['label']]=Path(v['path'])
    bounded_record=Path(__file__).with_name('corrosion-transport-bounded-validation.json')
    bounded_label='bounded/kks_raw_edges/l1/seed7'
    RUNS[bounded_label]=ROOT/'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed7'
    continuation_record=Path(__file__).with_name('corrosion-transport-continuation.json')
    for seed in (42,7):
        RUNS[f'phase/kks_raw_edges/l1/seed{seed}/rmsafter4000']=ROOT/f'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed{seed}-rmsafter4000'
    assert len(RUNS)==15
    baseline=RUNS['phase/physical_difference/l1/seed42']
    base_config=json.loads((baseline/'config.json').read_text())
    assert 'lr_tail_start' not in base_config['experiment']
    base_config['experiment'].update(lr_tail_start=0,lr_tail_end=1e-5,ch_rms_after=0)
    basecfg=flat(base_config)
    snapshots=json.loads((baseline/'source/files.json').read_text())
    sources={p:(baseline/'source'/p).read_bytes() for p in snapshots}
    initial_by_seed={42:(baseline/'model_initial.eqx').read_bytes(),7:(RUNS['prior/phase/s_bc/seed7']/'model_initial.eqx').read_bytes()}
    out={'scope':'Transport evaluation/norm screen, historical normalized controls and explicitly completed validation; independent CPU full-grid audit','runs':{}}
    histories={};predictions={};later_source_bytes={}
    prefix={}
    for seed in (42,7):
        original=RUNS[f'phase/kks_raw_edges/l1/seed{seed}']
        tail=RUNS[f'phase/kks_raw_edges/l1/seed{seed}/tail3000']
        checks={name:(original/name).read_bytes()==(tail/name).read_bytes() for name in
                ('model_initial.eqx','ckpt_001000.eqx','ckpt_002000.eqx','ckpt_003000.eqx')}
        loss_rows=lambda run:[json.loads(x) for x in (run/'loss_history.jsonl').read_text().splitlines() if json.loads(x)['step']<=3000]
        evals=[]
        for run in (original,tail):
            rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if json.loads(x)['step']<=3000]
            for row in rows:row.pop('elapsed_seconds')
            evals.append(rows)
        checks['loss_weights_prefix_exact']=loss_rows(original)==loss_rows(tail)
        checks['evaluation_prefix_exact_excluding_elapsed_time']=evals[0]==evals[1]
        prefix[str(seed)]={'checks':checks,'all_exact':all(checks.values())}
        print('Pre-tail trajectory check',seed,prefix[str(seed)],flush=True)
    out['pre_tail_trajectory_comparison']=prefix
    for label,run in RUNS.items():
        if label==bounded_label:
            # Permit CPU checks of the other12 models while this last GPU
            # control finishes; the final output still requires its success.
            while not bounded_record.exists():time.sleep(10)
            bounded_validation=json.loads(bounded_record.read_text())
            assert bounded_validation['returncode']==0
            assert bounded_validation['label']==label and Path(bounded_validation['path'])==run
        if '/rmsafter4000' in label:
            while True:
                if continuation_record.exists():
                    completed_continuation=json.loads(continuation_record.read_text())
                    if len(completed_continuation)==2:break
                time.sleep(10)
            assert all(v['returncode']==0 for v in completed_continuation)
            matching=[v for v in completed_continuation if v['label']==label]
            assert len(matching)==1 and Path(matching[0]['path'])==run
        c=ConfigDict(json.loads((run/'config.json').read_text()))
        prior=label.startswith('prior/')
        later='lr_tail_start' in c.experiment
        continuation='ch_rms_after' in c.experiment
        if prior:
            # Explicit schema adapter for the known archived legacy-L1 runs.
            for name in ('transport_evaluation','ch_norm','ch_rms_floor'):
                if name in c.experiment:raise ValueError('Unexpected historical schema')
            c.experiment.transport_evaluation='physical_difference'
            c.experiment.ch_norm='l1';c.experiment.ch_rms_floor=1e-12
        if not later:
            assert label in {f'{f}/{e}/{n}/seed42' for f,e,n in PLAN} or prior
            c.experiment.lr_tail_start=0;c.experiment.lr_tail_end=1e-5
        else:
            assert VALIDATION_RECORD.exists()
        if not continuation:c.experiment.ch_rms_after=0
        else:assert '/rmsafter4000' in label and c.experiment.ch_rms_after==4000
        cfg=flat(c.to_dict())
        allowed={'log.run_name','experiment.formulation','experiment.transport_evaluation','experiment.ch_norm','experiment.lr_tail_start','experiment.lr_tail_end','experiment.ch_rms_after','train.seed'}
        differences=[k for k in basecfg.keys()|cfg.keys() if k not in allowed and (k not in basecfg or k not in cfg or basecfg[k]!=cfg[k])]
        if differences:raise ValueError(f'Unmatched configs {label}:{differences}')
        assert (run/'model_initial.eqx').read_bytes()==initial_by_seed[c.train.seed]
        changed_sources=[]
        for path,source in sources.items():
            if (run/'source'/path).read_bytes()!=source:changed_sources.append(path)
        expected_changed={'examples/corrosion2d/residuals.py','examples/corrosion2d/residual_experiment.py','examples/corrosion2d/configs/residual_pilot.py'}
        later_changed={'examples/corrosion2d/residual_experiment.py','examples/corrosion2d/configs/residual_pilot.py'}
        assert set(changed_sources)==(expected_changed if prior else later_changed if later else set()),(label,changed_sources)
        saved_names=set(json.loads((run/'source/files.json').read_text()))
        helper='examples/corrosion2d/transport_optimizer.py'
        assert saved_names==set(sources)|({helper} if later else set())
        if later:
            assert (run/'source'/helper).is_file()
            this_source={name:(run/'source'/name).read_bytes() for name in saved_names}
            source_group='continuation' if continuation else 'tail_only'
            if source_group not in later_source_bytes:later_source_bytes[source_group]=this_source
            else:assert this_source==later_source_bytes[source_group],'Later cohort source drift'
            if len(later_source_bytes)==2:
                for name in this_source:
                    if name not in later_changed:
                        assert later_source_bytes['continuation'][name]==later_source_bytes['tail_only'][name],name
            import ast
            original=ast.parse((baseline/'source/examples/corrosion2d/residual_experiment.py').read_text())
            current=ast.parse((run/'source/examples/corrosion2d/residual_experiment.py').read_text())
            for name in ('nondim','graph_matrix','build_model','make_step','field_metrics','boundary_values','training_initial_data','training_boundary_data'):
                a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
                b=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
                assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),(label,name)
            for tree in (original,current):
                train_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
                loop=next(n for n in train_node.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
                if tree is original:original_loop=ast.dump(loop,include_attributes=False)
                else:
                    if continuation:
                        import copy
                        loop=copy.deepcopy(loop)
                        switches=[n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='current_step']
                        assert len(switches)==1
                        value=switches[0].value
                        expected=ast.parse('final_step if config.experiment.ch_rms_after and iteration >= config.experiment.ch_rms_after else step',mode='eval').body
                        assert ast.dump(value,include_attributes=False)==ast.dump(expected,include_attributes=False)
                        loop.body.remove(switches[0])
                        class RestoreStep(ast.NodeTransformer):
                            def visit_Name(self,node):
                                if node.id=='current_step':node.id='step'
                                return node
                        loop=RestoreStep().visit(loop)
                    assert ast.dump(loop,include_attributes=False)==original_loop,'Training-loop/state-flow drift'
            profile=json.loads((run/'summary.json').read_text())['learning_rate_profile']
            assert profile['tail_start']==c.experiment.lr_tail_start
            assert abs(profile['initial_rate']-c.train.lr_init)<1e-9
            target=c.experiment.lr_tail_end if c.experiment.lr_tail_start else c.train.lr_init*c.train.lr_decay_rate**((c.train.iterations-1)/c.train.lr_decay_every)
            if c.experiment.lr_tail_start:
                base_at=c.train.lr_init*c.train.lr_decay_rate**(c.experiment.lr_tail_start/c.train.lr_decay_every)
                target=base_at*(c.experiment.lr_tail_end/base_at)**((c.train.iterations-1-c.experiment.lr_tail_start)/(c.train.iterations-c.experiment.lr_tail_start))
            np.testing.assert_allclose(profile['last_update_rate'],target,rtol=2e-6)
        summary=json.loads((run/'summary.json').read_text())
        if continuation:
            assert summary['ch_rms_after']==4000 and summary['final_ch_norm']=='rms'
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
            _,trained_ch=compute_transport_pde(m,x,jnp.ones(1),sub,nd,c.experiment.formulation,.5,
                            transport_evaluation=c.experiment.transport_evaluation,bounded_eta=c.experiment.bounded_eta)
            return raw,m.constraint(raw),br,m.constraint(br),deriv[0,:,1],native_ch[:,0],trained_ch[:,0]
        raw,replay,br,bfields,ct,native_ch,trained_ch=(np.asarray(v) for v in evaluate(net))
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
            'trained_evaluation_full_CH_rms_free':rms(trained_ch[free]),
            'reference_terminal_mean_c_t_free':float(np.mean(-nd[0]*targetdq[-1,free])),
            'predicted_terminal_mean_c_t_free':float(np.mean(ct[free])),
            'stable_terminal_mean_CH_residual_free':float(np.mean((ct+nd[0]*dq[-1])[free])),
            'terminal_zero_cL_derivative_fraction':float(np.mean(np.asarray(jnp.tanh(jnp.asarray(raw[-1,:,1])))**2==1)),
            'terminal_cL_over_cle_range':[float(cl[-1].min()/cle),float(cl[-1].max()/cle)],
        }
        err=dq[-1]-targetdq[-1]
        interface=free&(targetp[-1]>.05)&(targetp[-1]<.95)
        near=free&(cKDTree(points[zero|one]).query(points)[0]<=c.nl.delta+1e-7)
        total=float(np.sum(err[free]**2))
        if total<=0:raise ValueError('Unexpected identically zero error; update spatial audit')
        top=max(1,int(np.ceil(.01*free.sum())))
        nodeids=np.flatnonzero(free);worst=nodeids[np.argsort(abs(err[free]))[-10:][::-1]]
        spatial={'top_1_percent_squared_Dq_error_fraction':float(np.sum(np.sort(err[free]**2)[-top:])/total),
                 'near_dirichlet_squared_Dq_error_fraction':float(np.sum(err[near]**2)/total),
                 'interface_Dq_rel_l2':rel(dq[-1,interface],targetdq[-1,interface]),
                 'max_abs_Dq_error':float(np.max(abs(err[free]))),
                 'worst_nodes':[{'xy':points[i].tolist(),'Dq_error':float(err[i]),
                     'predicted_s':float(cl[-1,i]/cle),'reference_s':float(targetcl[-1,i]/cle)} for i in worst]}
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
        result={'path':str(run),'configuration_initialization_sources_verified':True,'changed_sources_from_new_baseline':changed_sources,
                'transport_evaluation':c.experiment.transport_evaluation,'ch_norm':c.experiment.ch_norm,'lr_tail_start':c.experiment.lr_tail_start,'ch_rms_after':c.experiment.ch_rms_after,'data_loss_mode':c.experiment.data_loss_mode,
                'physical_metrics':fields,'transport':transport,'spatial':spatial,'boundary_sample_metrics':bcs,'initial_metrics':ic,
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
    norm_prefix={}
    for seed in (42,7):
        original=RUNS[f'phase/kks_raw_edges/l1/seed{seed}'];refined=RUNS[f'phase/kks_raw_edges/l1/seed{seed}/rmsafter4000']
        checks={name:(original/name).read_bytes()==(refined/name).read_bytes() for name in
                ('model_initial.eqx','ckpt_001000.eqx','ckpt_002000.eqx','ckpt_003000.eqx','ckpt_004000.eqx')}
        histories_pair=[];losses_pair=[]
        for directory in (original,refined):
            rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines() if json.loads(x)['step']<=4000]
            for row in rows:row.pop('elapsed_seconds')
            histories_pair.append(rows)
            losses_pair.append([json.loads(x) for x in (directory/'loss_history.jsonl').read_text().splitlines() if json.loads(x)['step']<=4000])
        checks['evaluation_prefix_exact_excluding_elapsed_time']=histories_pair[0]==histories_pair[1]
        checks['loss_weights_prefix_exact']=losses_pair[0]==losses_pair[1]
        norm_prefix[str(seed)]={'checks':checks,'all_exact':all(checks.values())}
    out['pre_RMS_trajectory_comparison']=norm_prefix
    path=Path(__file__);path.with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,key,title in zip(axes.ravel(),('phi','c','q','Dq_free'),('phase','total concentration','concentration drive q','transport Dq')):
        for label,history in histories.items():
            ax.semilogy([r['step'] for r in history],[r['terminal_rel_l2_'+key] for r in history],label=label.replace('physical_difference','legacy').replace('kks_raw_edges','raw'))
        ax.set(xlabel='Updates',ylabel='50 s relative L2',title=title);ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)
    labels=[label for label in predictions if not label.startswith('prior/')]
    rows=int(np.ceil(len(labels)/3))
    fig,axes=plt.subplots(rows,3,figsize=(13,3*rows),constrained_layout=True,squeeze=False)
    for ax,label in zip(axes.ravel(),labels):
        image=ax.pcolormesh(ref['X'],ref['Y'],predictions[label][-1],vmin=0,vmax=1,shading='nearest')
        ax.contour(ref['X'],ref['Y'],ref['phi'][-1],levels=[.5],colors='white',linewidths=.8)
        ax.set(title=label.replace('physical_difference','legacy').replace('kks_raw_edges','raw'),xlabel='x*',ylabel='y*',aspect='equal')
    for ax in axes.ravel()[len(labels):]:ax.set_visible(False)
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Phase at50 s; white=reference interface')
    fig.savefig(path.with_name(path.stem+'-fields.png'),dpi=150);plt.close(fig)

if __name__=='__main__':main()
