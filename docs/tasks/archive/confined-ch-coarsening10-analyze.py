"""Matched richer ten-wave T1.5 PINN replay and physical audits.

Requires the explicit nonlocal final CPU evaluate.py replay first. Local final
and both best-checkpoint replays run here. No reference/optimizer is changed.
"""
from pathlib import Path
import sys,json,shutil
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.evaluate import evaluate,field_plot
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.problem import boundary_samples

OUT=Path(__file__).parent
BASE=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
RUNS={'nonlocal':BASE/'20261027-coarsening10-nonlocal-01','local_ad':BASE/'20261027-coarsening10-local-ad-01'}


def selected_field_plot(ref,fields,path,labels):
    if len(ref['time'])!=21:raise ValueError('Archived ten-wave case expects21 saved times')
    indices=np.array([0,2,6,12,20]);view=dict(ref);view['time']=ref['time'][indices]
    field_plot(view,[field[indices] for field in fields],path,labels)


def plot_l2(histories,path):
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for name,h in histories.items():
        for ax,key in zip(axes,('pooled','terminal'),strict=True):
            ax.plot([p['step'] for p in h],[p[key+'_rel_l2_phi_local'] for p in h],label=name)
    for ax,title in zip(axes,('All saved times','Terminal t*=1.5'),strict=True):
        ax.axvline(5000,ls=':',color='gray');ax.set(xlabel='Completed optimizer iterations',ylabel='Phi relative L2',ylim=(0,1.08),title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Common refined FEM reference; SOAP5000 + L-BFGS500')
    fig.savefig(path,dpi=180);plt.close(fig)


def plot_normals(result,times,path):
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,color in [('nonlocal','C0'),('local_ad','C1')]:
        for wall,style in [('0','-'),('1','--'),('2',':')]:
            values=np.asarray(result['runs'][name]['replay']['wall_normal_rms'][wall])
            for j,ax in enumerate(axes):
                # Skip the shared exact-zero initial phi normal on the log axis.
                start=1 if j==0 else 0
                ax.semilogy(times[start:],np.maximum(values[start:,j],1e-14),style,color=color,label=name+' wall'+wall)
    for ax,label in zip(axes,['phi','mu'],strict=True):
        ax.set(xlabel='Time',ylabel='Local AD normal '+label+' RMS');ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle('Classical-normal diagnostic; native graph no-exchange is a different constraint',fontsize=10)
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('Use JAX_PLATFORMS=cpu for independent replay')
    configs={k:json.loads((p/'config.json').read_text()) for k,p in RUNS.items()}
    cleaned=json.loads(json.dumps(configs))
    for c in cleaned.values():del c['nl']['method'],c['log']['run_name']
    assert cleaned['nonlocal']==cleaned['local_ad']
    assert (RUNS['nonlocal']/'model_initial.eqx').read_bytes()==(RUNS['local_ad']/'model_initial.eqx').read_bytes()
    sources=[{str(p.relative_to(r/'source')):p.read_bytes() for p in (r/'source').rglob('*') if p.is_file()} for r in RUNS.values()]
    assert sources[0]==sources[1]
    c=ConfigDict(configs['nonlocal']);ref=load_reference(c);v=ref['volumes']
    A=native_matrix(ref['points'],v,c)
    checks=json.loads(Path(c.reference.path).with_suffix('.json').read_text())
    components,_=connected_components(A,directed=False)
    assert components==1
    weighted=sparse.diags(v)@A;asymmetry=float(np.max(np.abs((weighted-weighted.T).data),initial=0.))
    assert asymmetry<1e-12

    result=dict(status='passed',reference=checks,matching_initialized_bytes=True,
        matching_source_files=len(sources[0]),only_config_differences=['nl.method','log.run_name'],
        sampling_scope='same RNG key/code/shapes; actual entire GPU time trace not saved',runs={})
    result['operator_structure']=dict(points=len(v),physical_area=float(v.sum()),visible_pairs=(A.nnz-len(v))//2,connected_components=int(components),weighted_symmetry_defect=asymmetry,weighted_column_sum_defect=float(np.max(abs(np.asarray(v@A)))))
    predictions={};histories={}
    for name,run in RUNS.items():
        if name=='nonlocal':
            replay=json.loads((run/'replay.json').read_text())
            if Path(replay['checkpoint'])!=run/'model_final.eqx':raise ValueError('Need nonlocal final replay first')
        else:replay=evaluate(run,run/'model_final.eqx')
        with np.load(run/'replayed_prediction.npz') as f:fields={k:f[k].copy() for k in f.files}
        predictions[name]=fields
        summary=json.loads((run/'summary.json').read_text())
        row=dict(replay=replay,summary=summary,best=json.loads((run/'best_metrics.json').read_text()),native={})
        # Best selection may be the untrained step0; record it rather than call
        # an initialized checkpoint a successful trained model.
        cc=ConfigDict(configs[name]);key,_=jax.random.split(jax.random.PRNGKey(cc.train.seed))
        best_net=eqx.tree_deserialise_leaves(run/'model_best.eqx',_build_network(cc,key))
        initial_mean=jnp.dot(jnp.asarray(v),initial_field(jnp.asarray(ref['points']),cc,xp=jnp))/v.sum()
        from examples.confined_ch2d.network import predict_fields
        @eqx.filter_jit
        def best_fields(model):
            return jax.vmap(lambda t:predict_fields(model,jnp.asarray(ref['points']),t,initial_mean,jnp.asarray(v)))(jnp.asarray(ref['time']))
        best_prediction=np.asarray(best_fields(best_net));best_replay={}
        with np.load(run/'prediction_best.npz') as saved:
            for i,k in enumerate(('phi','mu')):
                np.testing.assert_allclose(best_prediction[...,i],saved[k],atol=1e-8,rtol=1e-8)
                best_replay[k]=float(np.max(abs(best_prediction[...,i]-saved[k])))
                np.testing.assert_allclose(relative(best_prediction[...,i],ref[k+'_local'],v),row['best']['pooled_rel_l2_'+k+'_local'],atol=1e-8,rtol=1e-8)
        row['best_cpu_replay_max_absolute']=best_replay
        row['best_is_untrained_initialization']=row['best']['step']==0
        row['best_phi_fields_scope']='Validation-selected checkpoint; step0, if selected, is untrained'
        selected_field_plot(ref,[ref['phi_local'],best_prediction[...,0],np.abs(best_prediction[...,0]-ref['phi_local'])],OUT/('confined-ch-coarsening10-'+name+'-best-phi-fields.png'),['Common FEM reference',f'Best checkpoint, step{row["best"]["step"]}','Absolute error'])
        kind='nonlocal' if name=='nonlocal' else 'local'
        for key,a in fields.items():
            row['native'][key]=dict(pooled=relative(a,ref[key+'_'+kind],v),terminal=relative(a[-1],ref[key+'_'+kind][-1],v))
            np.testing.assert_allclose(row['native'][key]['pooled'],summary['final_metrics']['pooled_rel_l2_'+key],atol=1e-8,rtol=1e-8)
            row['replay']['common_local'][key]['positive_time_pooled']=relative(a[1:],ref[key+'_local'][1:],v)
        initial=initial_field(ref['points'],c,xp=np)
        row['prescribed_mean']=float(initial@v/v.sum())
        def correlation_to_ic(phi):
            a=phi-float(phi@v/v.sum());b=initial-row['prescribed_mean']
            denominator=np.sqrt(np.sum(v*a*a)*np.sum(v*b*b))
            if denominator<=0:raise ValueError('IC correlation requires nonconstant phase')
            return float(np.sum(v*a*b)/denominator)
        row['terminal_structure']=dict(
            prediction_change_from_ic=relative(fields['phi'][-1],initial,v),
            reference_change_from_ic=relative(ref['phi_local'][-1],initial,v),
            prediction_ic_correlation=correlation_to_ic(fields['phi'][-1]),
            reference_ic_correlation=correlation_to_ic(ref['phi_local'][-1]),
            prediction_pure_fraction=float(v[abs(fields['phi'][-1])>.9].sum()/v.sum()),
            reference_pure_fraction=float(v[abs(ref['phi_local'][-1])>.9].sum()/v.sum()))
        row['local_reference_initial_mean_offset']=float(ref['phi_local'][0]@v/v.sum()-row['prescribed_mean'])
        row['local_reference_quadrature_mass_drift']=float(np.max(abs(ref['phi_local']@v/v.sum()-ref['phi_local'][0]@v/v.sum())))
        assert replay['mass_max_drift']<1e-12
        band_mu=np.abs(ref['phi_local'])<.5
        row['interface_phi_local']=dict(definition='abs(phi_local)<0.5',pooled=relative(fields['phi'][band_mu],ref['phi_local'][band_mu],np.broadcast_to(v,band_mu.shape)[band_mu]),terminal=relative(fields['phi'][-1,band_mu[-1]],ref['phi_local'][-1,band_mu[-1]],v[band_mu[-1]]))
        if name=='nonlocal':
            phi=fields['phi'];energy=np.sum(v*(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*(A@phi.T).T),axis=1)
        else:
            cc=ConfigDict(configs[name]);key,_=jax.random.split(jax.random.PRNGKey(cc.train.seed))
            net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(cc,key))
            @eqx.filter_jit
            def gradients(model):
                return jax.vmap(lambda t:jax.vmap(jax.jacfwd(model,argnums=0),in_axes=(0,None))(jnp.asarray(ref['points']),t))(jnp.asarray(ref['time']))
            grad=np.asarray(gradients(net))[:,:,0,:]
            phi=fields['phi'];energy=np.sum(v*(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*np.sum(grad*grad,axis=-1)),axis=1)
        row['native_energy']=energy.tolist();row['sampled_max_energy_increase']=max(0.,float(np.max(np.diff(energy))))
        histories[name]=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        # Diagnose initial vs positive-time fitting without altering the trained model.
        cc=ConfigDict(configs[name]);key,_=jax.random.split(jax.random.PRNGKey(cc.train.seed))
        net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(cc,key))
        b,n,bw,_=boundary_samples(cc)
        operator=RegionalLaplacian(ref['points'],v,cc) if name=='nonlocal' else None
        inputs=(jnp.asarray(ref['points']),jnp.asarray(v),operator,jnp.asarray(b),jnp.asarray(n),jnp.asarray(bw))
        prescribed=jnp.asarray(row['prescribed_mean'])
        initial_loss=eqx.filter_jit(make_loss_fn(*inputs,cc))
        _,initial_parts=initial_loss(net,jnp.zeros(0,dtype=jnp.float64),prescribed)
        positive_config=ConfigDict(cc.to_dict());positive_config.sample.include_initial_time=False
        positive_loss=eqx.filter_jit(make_loss_fn(*inputs,positive_config))
        _,positive_parts=positive_loss(net,(jnp.arange(16,dtype=jnp.float64)+.5)*c.physical.terminal_time/16,prescribed)
        row['fixed_residual_parts']=dict(order=['phase','chemical']+(['normal_phi','normal_mu'] if name=='local_ad' else []),
            initial=np.asarray(initial_parts).tolist(),positive_midpoints=np.asarray(positive_parts).tolist())
        weights=np.array([cc.loss.phase_weight,cc.loss.chemical_weight]+([cc.loss.boundary_phi_weight,cc.loss.boundary_mu_weight] if name=='local_ad' else []))
        r0=np.asarray(initial_parts)*weights;rp=np.asarray(positive_parts)*weights
        denominator=float(r0.sum()+16*rp.sum())
        row['fixed17_initial_loss_fraction']=float(r0.sum()/denominator)
        row['fixed17_initial_chemical_loss_fraction']=float(r0[1]/denominator)
        row['boundary_scope']='Local AD normal RMS diagnostics; not exact native nonlocal Neumann constraints'
        result['runs'][name]=row
        for file in ('config.json','summary.json','best_metrics.json','metrics.jsonl','loss.jsonl','refinement_summary.json','replay.json','phi_fields.png','convergence.png'):
            shutil.copyfile(run/file,OUT/('confined-ch-coarsening10-'+name+'-'+file))
    selected_field_plot(ref,[ref['phi_local'],predictions['nonlocal']['phi'],predictions['local_ad']['phi']],OUT/'confined-ch-coarsening10-common-fields.png',['Common local reference','Nonlocal PINN','Local AD PINN'])
    selected_field_plot(ref,[ref['phi_nonlocal'],ref['phi_local'],np.abs(ref['phi_nonlocal']-ref['phi_local'])],OUT/'confined-ch-coarsening10-reference-gap.png',['Native nonlocal reference','Local FEM reference','Absolute model/grid error'])
    fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
    for name,h in histories.items():
        axes[0].semilogy([p['step'] for p in h],[p['pooled_rel_l2_phi_local'] for p in h],label=name)
        axes[1].semilogy([p['step'] for p in h],[p['terminal_rel_l2_phi_local'] for p in h],label=name)
        axes[2].plot(ref['time'],[relative(a,b,v) for a,b in zip(predictions[name]['phi'],ref['phi_local'],strict=True)],label=name)
        axes[3].plot(ref['time'],result['runs'][name]['native_energy'],label=name+' PINN')
    for kind in ('local','nonlocal'):axes[3].plot(ref['time'],ref['energy_'+kind],':',label=kind+' reference')
    axes[0].set(xlabel='Optimizer iterations',ylabel='Pooled phi relative L2, common local target')
    axes[1].set(xlabel='Optimizer iterations',ylabel='Terminal phi relative L2, common FEM')
    axes[2].set(xlabel='Time',ylabel='Phi relative L2, common local target')
    axes[3].set(xlabel='Time',ylabel='Each native free-energy functional')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.savefig(OUT/'confined-ch-coarsening10-comparison.png',dpi=180);plt.close(fig)
    plot_l2(histories,OUT/'confined-ch-coarsening10-l2-convergence.png')
    result['training_outcome']='phase_learning_failed_at_declared_budget'
    result['scope']='Single seed matched-budget nonlocal/local AD comparison with explicit local Neumann penalties, no local spectral. Replay/parity audit success is not PDE solution accuracy: both best checkpoints are untrained step0, positive-time fields remain inaccurate and sampled PINN energies increase. Finite horizon/geometry/quadrature bias is distinct from PINN fitting. No stability/statistical claim or pointwise classical Neumann guarantee at finite delta.'
    plot_normals(result,ref['time'],OUT/'confined-ch-coarsening10-boundary-normals.png')
    (OUT/'confined-ch-coarsening10-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
