"""Saved-metric comparison of5000and20000cold budgets on identical geometries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compare(run,short_run,output):
    output.mkdir(parents=True,exist_ok=False)
    runs=(short_run,run)
    histories=[[json.loads(line) for line in (r/'metrics.jsonl').read_text().splitlines()] for r in runs]
    metrics=[json.loads((r/'evaluation/metrics.json').read_text()) for r in runs]
    summaries=[json.loads((r/'summary.json').read_text()) for r in runs]
    effects=[json.loads((r/'geometry_effect/metrics.json').read_text()) for r in runs]
    assert effects[0]['common_points']==effects[1]['common_points']
    assert effects[0]['reference']==effects[1]['reference']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for h,label,color in zip(histories,('5000-step cold run','20000-step cold run'),('C1','C0'),strict=True):
        for ax,key in zip(axes,('phi_pooled','phi_terminal'),strict=True):
            for split,style in (('train','--'),('test','-')):
                ax.semilogy([r['step'] for r in h],[100*r[split+'_'+key] for r in h],
                    color=color,linestyle=style,marker='o',markersize=2.5,
                    label=label+(': 16 train monitors' if split=='train' else ': 8 validation domains'))
            ax.set(xlabel='Adam updates',ylabel='Phase relative L2 (%)',
                title='All saved times' if key=='phi_pooled' else 'Terminal time t = 0.75')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'convergence.png',dpi=180);fig.savefig(output/'convergence.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
    for r,label,color in zip(runs,('5000-step cold run','20000-step cold run'),('C1','C0'),strict=True):
        loss=[json.loads(line) for line in (r/'loss.jsonl').read_text().splitlines()]
        ax.semilogy([a['step'] for a in loss],[a['loss'] for a in loss],color=color,alpha=.8,label=label)
    ax.set(xlabel='Adam updates',ylabel='Sampled preconditioned PDE loss');ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'physics_loss.png',dpi=180);plt.close(fig)
    history=histories[1];tail=[r for r in history if r['step']>=15000]
    trends={}
    for key in ('train_phi_pooled','test_phi_pooled','test_phi_terminal','test_mu_pooled'):
        x=np.asarray([r['step'] for r in tail]);y=np.asarray([r[key] for r in tail])
        trends[key]=dict(tail_start=15000,tail_end=20000,final=float(y[-1]),
            minimum=float(y.min()),maximum=float(y.max()),
            linear_slope_percentage_points_per1000steps=float(np.polyfit((x-x[0])/1000,100*y,1)[0]))
    milestones=[{k:r[k] for k in ('step','train_phi_pooled','test_phi_pooled','test_phi_terminal','test_mu_pooled')}
        for r in history if r['step'] in (5000,10000,15000,20000)]
    paired=[]
    for index in range(512,520):
        rows=[next(row for row in m['conditions'] if row['index']==index) for m in metrics]
        paired.append(dict(index=index,short_phi_pooled=rows[0]['phi_pooled'],long_phi_pooled=rows[1]['phi_pooled'],
            short_phi_terminal=rows[0]['phi_terminal'],long_phi_terminal=rows[1]['phi_terminal']))
    report=dict(run=str(run),short_run=str(short_run),same512training_and8validation_domains=True,
        short_native_test=metrics[0]['test'],long_native_test=metrics[1]['test'],
        short_local_FEM=metrics[0]['local_FEM'],long_local_FEM=metrics[1]['local_FEM'],
        geometry_response_relative_error=[e['geometry_response_relative_error'] for e in effects],
        geometry_response_amplitude=[e['geometry_response_amplitude_ratio'] for e in effects],
        geometry_response_correlation=[e['geometry_response_correlation'] for e in effects],
        cold_trainer_seconds=[s['wall_seconds'] for s in summaries],
        warm_update_seconds=[s['mean_update_seconds_after_100'] for s in summaries],
        milestones=milestones,tail_trends=trends,paired_validation=paired,
        selection='Fixed20000step final; tail extrema are diagnostics, not checkpoint selection',
        scope='One matched cold seed and8reused development domains; native/local operator bias remains')
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--short-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.run,a.short_run,a.output)
