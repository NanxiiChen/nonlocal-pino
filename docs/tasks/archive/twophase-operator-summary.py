"""Archive-quality comparison and scientific figures from direct saved paths."""
from pathlib import Path
import json
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STAGES=['preconditioning','refinement','resolution','conserved','initial-time','confirmation','short-confirmation']
OUT=ROOT/'reports/twophase-operator-summary'


def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    OUT.mkdir(exist_ok=True)
    gpu={}
    for stage in ['conserved','short-confirmation']:
        verification=json.loads((ROOT/('reports/twophase-operator-'+stage)/'gpu-verification.json').read_text())
        if verification['status']!='passed':raise ValueError('Incomplete frozen GPU verification')
        for item in verification['runs']:
            if item['run'] in gpu:raise ValueError('Duplicate verified run')
            gpu[item['run']]=item
    rows=[];histories={};predictions={}
    for stage in STAGES:
        report=ROOT/('reports/twophase-operator-'+stage)
        manifest=json.loads((report/'manifest.json').read_text())
        analysis=json.loads((report/'analysis.json').read_text())
        if manifest['status']!='completed' or analysis['status']!='passed':raise ValueError('Incomplete '+stage)
        for item in manifest['runs']:
            run=Path(item['run']);c=json.loads((run/'config.json').read_text())
            s=json.loads((run/'summary.json').read_text())
            with np.load(run/'prediction.npz') as f:phi=f['phi'];q=f['q']
            with np.load(c['reference']['path']) as f:
                target=f['phi_local'];matched=f['phi_nonlocal'];qt=f['q_nonlocal'];time=f['time']
            if str(run) not in gpu:raise ValueError('Missing frozen GPU audit')
            metrics=s['final_metrics']
            np.testing.assert_allclose(relative(phi,target),metrics['pooled_rel_l2_phi_local'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(relative(phi[-1],target[-1]),metrics['terminal_rel_l2_phi_local'],atol=1e-12,rtol=0)
            refinement=s['refinement'] if 'refinement' in s else None
            row=dict(stage=stage,arm=item['arm'],run=str(run),seed=c['train']['seed'],
                config_path=item['command'][item['command'].index(next(v for v in item['command'] if v.startswith('--config=')))][9:],
                phase_map=c['phase']['map'] if 'phase' in c else 'additive',
                residual_metric=c['preconditioner']['kind'] if 'preconditioner' in c else c['residual']['mode'] if 'residual' in c else 'original',
                advection=c['advection'] if 'advection' in c else 'AD_point_gradient',
                exact_mass='phase_mass' in c,initial_pde_time='include_initial_time' in c['sample'] and c['sample']['include_initial_time'],
                soap_updates=c['train']['iterations'],refinement_iterations=refinement['iterations'] if refinement else 0,
                dtype=c['train']['compute_dtype'],pooled_phi_relative_L2_local=relative(phi,target),
                terminal_phi_relative_L2_local=relative(phi[-1],target[-1]),
                pooled_phi_relative_L2_matched=relative(phi,matched),terminal_phi_relative_L2_matched=relative(phi[-1],matched[-1]),
                pooled_interface_phi_relative_L2_local=relative(phi[np.abs(target)<.9],target[np.abs(target)<.9]),
                pooled_q_rms_matched=float(np.sqrt(np.mean((q-qt)**2))),
                terminal_q_rms_matched=metrics['terminal_q_rms_error'],
                max_soft_area_drift=metrics['gas_area_relative_drift_max'],wall_seconds=s['wall_seconds'],
                periodic_boundary_verified=gpu[str(run)]['periodic_frequencies_fixed'])
            rows.append(row);histories[(stage,item['arm'])]=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            predictions[(stage,item['arm'])]=phi
    if len(rows)!=16 or len(gpu)!=16:raise ValueError('Expected all16 completed/audited controls')
    selected=next(r for r in rows if r['stage']=='conserved' and r['arm']=='conserved')
    independent=next(r for r in rows if r['stage']=='short-confirmation')
    # The slight pooled-only endpoint gain does not outweigh worse terminal/q
    # and extra collocation; the original residual is preferred over causal.
    comparison=dict(status='passed',selection='simple physically consistent original-residual candidate; all outcomes retained',
        recommended_config='examples/twophase2d/configs/train_vortex_conserved.py',selected=selected,
        independent_seed_confirmation=independent,
        reference_gap=dict(pooled=relative(matched,target),terminal=relative(matched[-1],target[-1])),
        limitations=['one prescribed-flow CH case; no solved Navier-Stokes',
            'legacy pre-fix periodic frequencies drifted',
            'one resolved pair has inactive-wrapper source differences',
            'longer fixed-collocation L-BFGS lowers loss without guaranteed lower field error',
            'grid conservation is enforced; field/interface errors remain measured'],rows=rows)
    (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    controls=[('conserved','consistent','Additive, original residual'),('conserved','conserved','Cubic, original, seed42'),('short-confirmation','seed2026','Cubic, original, seed2026')]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for stage,arm,label in controls:
        h=histories[(stage,arm)]
        axes[0].semilogy([r['step'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=label)
        axes[1].semilogy([r['step'] for r in h],[r['terminal_rel_l2_phi_local'] for r in h],label=label)
        phi=predictions[(stage,arm)]
        axes[2].plot(time,[relative(a,b) for a,b in zip(phi,target,strict=True)],label=label)
    axes[0].set(xlabel='SOAP then L-BFGS iterations',ylabel='Pooled phi relative L2, local reference')
    axes[1].set(xlabel='SOAP then L-BFGS iterations',ylabel='Terminal phi relative L2, local reference')
    axes[2].set(xlabel='Physical time',ylabel='Final-model phi relative L2, local reference')
    for ax in axes:
        ax.axhline(.01,color='black',ls=':',lw=1);ax.grid(alpha=.2)
    for ax in axes[:2]:ax.axvline(5000,color='gray',ls='--',lw=.8)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/'convergence.png',dpi=180);plt.close(fig)
    phi=predictions[('conserved','conserved')];error=np.abs(phi-target)
    ids=[0,5,10,15,20];n=phi.shape[-1];axis=np.arange(n+1)/n
    vmin=min(-1.,float(phi.min()),float(target.min()));vmax=max(1.,float(phi.max()),float(target.max()))
    fig,axes=plt.subplots(3,5,figsize=(14,8),layout='constrained')
    for row,field in enumerate([target,phi,error]):
        for col,i in enumerate(ids):
            ax=axes[row,col];padded=np.pad(field[i],((0,1),(0,1)),mode='wrap')
            image=ax.pcolormesh(axis,axis,padded,shading='nearest',cmap='viridis' if row<2 else 'magma',vmin=vmin if row<2 else 0.,vmax=vmax if row<2 else float(error[ids].max()))
            ax.set(aspect='equal',xlim=(0,1),ylim=(0,1))
            if row<2:ax.contour(axis,axis,padded,levels=[0],colors='white',linewidths=.6)
            if not row:ax.set_title(f't={time[i]:.2f} (t/T={time[i]/time[-1]:.2f})')
            if col==0:ax.set_ylabel(['Local reference','PINN: cubic/original','Absolute error'][row]+'\ny')
            if row==1:
                phase_image=image
                ax.text(.02,.02,f'L2={100*relative(phi[i],target[i]):.2f}%',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
            if row==2:ax.set_xlabel('x')
    fig.colorbar(phase_image,ax=axes[:2].ravel().tolist(),label='phi, full field range',shrink=.75)
    fig.colorbar(image,ax=axes[2].ravel().tolist(),label='Absolute error',shrink=.75)
    fig.suptitle('Prescribed reversed-vortex CH; SOAP5000 + L-BFGS1000; common local reference')
    fig.savefig(OUT/'recommended-phi.png',dpi=180);fig.savefig(OUT/'recommended-phi.svg');plt.close(fig)
    np.savez_compressed(OUT/'recommended-fields.npz',time=time,reference=target,prediction=phi,absolute_error=error)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,arm in zip(axes,['seed42','seed2026'],strict=True):
        row=next(r for r in rows if r['stage']=='confirmation' and r['arm']==arm)
        run=Path(row['run']);loss=[json.loads(line) for line in (run/'refinement.jsonl').read_text().splitlines()]
        h=[r for r in histories[('confirmation',arm)] if r['step']>=5000]
        a=ax.plot([r['step']-5000 for r in h],[100*r['pooled_rel_l2_phi_local'] for r in h],color='tab:blue',label='Pooled phi L2')
        other=ax.twinx();b=other.semilogy([r['iteration'] for r in loss],[r['loss'] for r in loss],color='tab:red',label='Training PDE loss')
        ax.set(xlabel='L-BFGS iterations',ylabel='Pooled phi relative L2 (%)',title=arm)
        other.set_ylabel('Training PDE loss');ax.legend(a+b,[v.get_label() for v in a+b],fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'residual-vs-error.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(selected=selected,independent=independent),indent=2),flush=True)

if __name__=='__main__':main()
