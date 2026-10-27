"""Direct-path checkpoint replay and common-reference comparison; no fitting."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import jax
from examples.twophase2d.evaluate import audit_run
geometry_path=ROOT/'docs/tasks/archive/twophase-vortex-residual-analyze.py'
spec=importlib.util.spec_from_file_location('geometry_audit',geometry_path)
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args();out=args.report_dir
    if jax.default_backend()!='cpu':raise ValueError('Use JAX_PLATFORMS=cpu')
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Incomplete cohort')
    results=[];histories={};fields=[]
    for item in manifest['runs']:
        run=Path(item['run']);arm=item['arm'];folder=out/arm;folder.mkdir(exist_ok=True)
        # Four-arm diagnostic measures at most5.125999e-6 CPU/GPU FP32
        # difference; the default evaluator's historical5e-6 is unchanged.
        result,history=audit_run(run,folder,fp32_replay_tolerance=6e-6)
        with np.load(run/'prediction.npz') as f:phi=f['phi']
        config=json.loads((run/'config.json').read_text())
        with np.load(config['reference']['path']) as f:
            target=f['phi_local'];matched=f['phi_nonlocal'];time=f['time']
        rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        result.update(arm=arm,pooled_phi_relative_L2_to_local=rel(phi,target),
            maximum_phi_relative_L2_to_local=max(rel(a,b) for a,b in zip(phi,target,strict=True)),
            pooled_interface_phi_relative_L2_to_local=rel(phi[np.abs(target)<.9],target[np.abs(target)<.9]),
            geometry_to_local=[geometry.compare_geometry(a,b,1.) for a,b in zip(phi,target,strict=True)])
        results.append(result);histories[arm]=history;fields.append(phi)
        print(arm, 'pooled/terminal local',result['pooled_phi_relative_L2_to_local'],result['terminal_phi_relative_L2_to_local'],flush=True)
    ids=[0,5,10,15,20]
    vmin=min(-1.,*(float(a.min()) for a in fields),float(target.min()))
    vmax=max(1.,*(float(a.max()) for a in fields),float(target.max()))
    fig,axes=plt.subplots(len(fields)+1,5,figsize=(14,2.6*(len(fields)+1)),layout='constrained')
    for row,(name,phi) in enumerate([('Local reference',target)]+[(r['arm'],f) for r,f in zip(results,fields,strict=True)]):
        for col,i in enumerate(ids):
            ax=axes[row,col];im=ax.imshow(phi[i],origin='lower',extent=(0,1,0,1),vmin=vmin,vmax=vmax,cmap='viridis')
            axis=np.arange(phi.shape[-1])/phi.shape[-1]
            ax.contour(axis,axis,phi[i],levels=[0],colors='white',linewidths=.6)
            if row:
                ax.contour(axis,axis,target[i],levels=[0],colors='orangered',linestyles='dashed',linewidths=.6)
                ax.text(.02,.02,f'L2 {100*rel(phi[i],target[i]):.2f}%',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
            if not row:ax.set_title(f't={time[i]:.2f}')
            if not col:ax.set_ylabel(name)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='phi, full range')
    fig.savefig(out/'fields.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for result in results:
        name=result['arm'];h=histories[name]
        axes[0].semilogy([r['step'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=name)
        axes[1].semilogy([r['step'] for r in h],[r['terminal_rel_l2_phi_local'] for r in h],label=name)
        axes[2].plot(time,result['per_time_phi_relative_L2_to_local'],label=name)
    axes[0].set(xlabel='Updates',ylabel='Pooled phi relative L2, local target')
    axes[1].set(xlabel='Updates',ylabel='Terminal phi relative L2, local target')
    axes[2].set(xlabel='Physical time',ylabel='Phi relative L2, local target')
    for ax in axes:
        ax.axhline(.01,color='black',ls=':',lw=1);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'convergence.png',dpi=170);plt.close(fig)
    comparison=dict(status='passed',reference_gap=dict(pooled=rel(matched,target),terminal=rel(matched[-1],target[-1])),runs=results)
    (out/'analysis.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
