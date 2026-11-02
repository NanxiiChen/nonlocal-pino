"""Plot complete recorded L2 and sampled normalized loss histories; no fitting."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(run,output):
    run,output=Path(run),Path(output)
    if not run.is_absolute() or not output.is_absolute():raise ValueError('Absolute run/output paths required')
    summary=json.loads((run/'summary.json').read_text())
    config=json.loads((run/'config.json').read_text())
    if summary['status']!='completed':raise ValueError('Complete training endpoint required')
    metrics=[json.loads(s)for s in (run/'metrics.jsonl').read_text().splitlines()]
    losses=[json.loads(s)for s in (run/'loss.jsonl').read_text().splitlines()]
    steps=np.array([r['step']for r in metrics]);updates=summary['iterations']
    np.testing.assert_array_equal(steps,np.arange(0,updates+1,config['log']['eval_every']))
    expected_loss_steps=np.r_[1,np.arange(100,updates+1,100)]
    loss_steps=np.array([r['step']for r in losses]);np.testing.assert_array_equal(loss_steps,expected_loss_steps)
    names=('phi','terminal_phi','velocity','pressure','dispersed_phase','chemical')
    for name in names:
        y=np.array([r[name]for r in metrics])
        if not np.isfinite(y).all() or np.any(y<=0):raise ValueError(f'Invalid relative L2 series{name}')
        np.testing.assert_allclose(y[-1],summary['metrics'][name],atol=2e-7,rtol=0)
    all_parts=tuple(losses[0]['parts'])
    for r in losses:
        if tuple(r['parts'])!=all_parts:raise ValueError('Loss-part schema changed during training')
        if any(not np.isfinite(v) or v<0 for v in [r['total'],*r['parts'].values()]):raise ValueError('Nonfinite or negative normalized MSE')
        np.testing.assert_allclose(r['total'],sum(r['parts'].values()),rtol=2e-6,atol=1e-7)
    output.mkdir(parents=True,exist_ok=False)
    colors=('#0072B2','#D55E00')
    specs=[(('phi','Phase field, all times'),('terminal_phi','Phase field, terminal time')),
           (('velocity','Velocity'),('pressure','Pressure')),
           (('dispersed_phase','Droplet fraction'),('chemical','Chemical potential'))]
    titles=('Phase-field error','Flow-field error','Droplet and chemical-potential error')
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.0),layout='constrained')
    for ax,series,title in zip(axes,specs,titles,strict=True):
        for (key,label),color in zip(series,colors,strict=True):
            y=100*np.array([r[key]for r in metrics])
            ax.semilogy(steps,y,'o-',ms=3.5,lw=1.6,color=color,label=f'{label}: {y[-1]:.2f}%')
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Relative L2 (%)',xlim=(0,updates))
        ax.grid(alpha=.2,which='both');ax.legend(fontsize=8,loc='upper right')
    fig.suptitle(f'Random-wall NS–CH: complete cold-start training (width {config["arch"]["width"]}, {config["train"]["optimizer"].upper()})\n'
                 'One development geometry; recorded FEM errors every 500 updates',fontsize=12)
    fig.savefig(output/'relative-l2.png',dpi=200);fig.savefig(output/'relative-l2.pdf');plt.close(fig)
    part=lambda key:np.array([r['parts'][key]for r in losses])
    boundary=('wall_phase','wall_mu','outlet_traction','outlet_phase','outlet_mu')
    grouped=dict(total=np.array([r['total']for r in losses]),phase=part('phase'),chemical=part('chemical'),
        momentum=part('momentum_x')+part('momentum_y'),boundary=sum(part(k)for k in boundary))
    fig,axes=plt.subplots(1,2,figsize=(12,4.0),layout='constrained')
    def draw(ax,key,values,label,**kwargs):
        if np.any(values<=0):raise ValueError(f'Positive values required for the log-loss display: {key}')
        ax.semilogy(loss_steps,values,label=label,lw=1.3,**kwargs)
    for key,label in (('total','Total'),('phase','CH transport'),('chemical','Chemical relation'),
                      ('momentum','Momentum x + y'),('boundary','All soft BC terms')):
        draw(axes[0],key,grouped[key],label)
    for key,label in zip(boundary,('Wall phase','Wall chemical potential','Outlet traction','Outlet phase','Outlet chemical potential'),strict=True):
        draw(axes[1],key,part(key),label)
    for ax,title in zip(axes,('Normalized sampled losses','Normalized boundary losses'),strict=True):
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Normalized MSE',xlim=(0,updates))
        ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
    fig.suptitle('Raw logged values; fresh geometry/time batches, no smoothing\n'
                 'These normalized sampled losses are distinct from fixed-geometry FEM errors',fontsize=11)
    fig.savefig(output/'sampled-losses.png',dpi=200);fig.savefig(output/'sampled-losses.pdf');plt.close(fig)
    report=dict(status='complete_recorded_histories_verified_and_plotted',run=str(run),updates=updates,
        L2_records=len(metrics),loss_records=len(losses),L2_steps=steps.tolist(),loss_steps=loss_steps.tolist(),
        last_L2={k:metrics[-1][k]for k in names},last_losses={k:float(v[-1])for k,v in grouped.items()},
        last_500_update_relative_improvement={k:float(1-metrics[-1][k]/metrics[-2][k])for k in names},
        scope='L2:one development geometry on fixed FEM queries; loss:actual stochastic physics batches, no evolving labels',
        temporal_scope='Phi/droplet includes all saved times; velocity/pressure/chemical exclude unprescribed initial pressure/time0',
        smoothing=False,curve_fitting=False,loss_grouping='Momentum x+y, sum of five BC MSE terms; all original scaling retained')
    (output/'convergence-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'curve-data.json').write_text(json.dumps(dict(metrics=metrics,losses=losses),indent=2,allow_nan=False)+'\n')
    (output/'plot-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.run,a.output)
