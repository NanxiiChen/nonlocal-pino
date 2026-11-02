"""Actual cold5k histories for the BDF/JVP by uniform/adaptive-loss controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(runs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    keys=('uniform_BDF','adaptive_BDF','uniform_JVP','adaptive_JVP')
    if set(runs)!=set(keys):raise ValueError('Exactly four named temporal/weighting controls required')
    configs={k:json.loads((Path(p)/'config.json').read_text())for k,p in runs.items()}
    baseline=configs['uniform_BDF'];normalized=[]
    for key in keys:
        c=json.loads(json.dumps(configs[key]))
        if c['train']['iterations']!=5000 or c['sample']['Nt']!=40:
            raise ValueError('Matched cold5000/Nt40 controls required')
        if 'continuation'in c['train']or 'initialization'in c['train']:
            raise ValueError('Curves require cold controls, not imported models')
        adaptive=key.startswith('adaptive');jvp=key.endswith('JVP')
        if adaptive!=('causal'in c)or jvp!=('time_derivative'in c):
            raise ValueError(f'Curve label does not match actual config: {key}')
        if adaptive:
            if 'adaptation'not in c['causal']:raise ValueError('Adaptive weighting required')
            del c['causal']
        if jvp:
            if c['time_derivative']!={'kind':'jvp','sampling':'continuous_stratified'}:
                raise ValueError('Continuous physicaltimeJVP required')
            del c['time_derivative'];c['time_discretization']=baseline['time_discretization']
        c['log']['run_name']=baseline['log']['run_name'];normalized.append(c)
    if any(c!=baseline for c in normalized):raise ValueError('Controls differ beyond timeoperator/sampling/causal/runname')
    if configs['adaptive_BDF']['causal']!=configs['adaptive_JVP']['causal']:
        raise ValueError('Adaptive controls use different feedback settings')
    histories={}
    for key in keys:
        lines=(Path(runs[key])/'metrics.jsonl').read_bytes().splitlines(keepends=True)
        rows=[json.loads(s)for s in lines if s.endswith(b'\n')]
        steps=np.array([r['step']for r in rows])
        if not len(rows)or steps[0]!=0 or np.any(np.diff(steps)<=0)or steps[-1]>5000:
            raise ValueError('Invalid actual metric history')
        histories[key]=rows
    common=min(rows[-1]['step']for rows in histories.values())
    if any(common not in [r['step']for r in rows]for rows in histories.values()):
        raise ValueError('No common latest evaluated update')
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    titles=('Phase: all snapshots','Phase: terminal t=6','Minority fraction: all snapshots',
        'Velocity: positive times','Pressure: positive times','Chemical potential: positive times')
    fields=('phi','terminal_phi','dispersed_phase','velocity','pressure','chemical')
    for ax,field,title in zip(axes.ravel(),fields,titles,strict=True):
        for key in keys:
            rows=histories[key];x=[r['step']for r in rows];y=100*np.array([r[field]for r in rows])
            if np.any(~np.isfinite(y))or np.any(y<=0):raise ValueError('Log-error curves require finite positive actual metrics')
            color='#0072B2'if key.endswith('BDF')else '#D55E00'
            style='--'if key.startswith('adaptive')else '-'
            ax.plot(x,y,color=color,ls=style,lw=2.3 if key=='uniform_JVP'else 1.4,label=key.replace('_',' '))
            if key=='uniform_JVP':ax.scatter(x[-1],y[-1],color=color,s=20,zorder=5)
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Relative L2 (%)',xlim=(0,5000),yscale='log')
        ax.yaxis.set_major_formatter(ScalarFormatter());ax.grid(alpha=.2,which='both')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Same cold model / geometry prior / physics; one development FEM geometry\n'
        'New uniform JVP shown only through its actual last monitor; minority curve is pooled, not terminal shape',fontsize=10)
    output.mkdir(parents=True);fig.savefig(output/'convergence.png',dpi=200)
    fig.savefig(output/'convergence.pdf');plt.close(fig)
    report=dict(status='actual_temporal_weighting_histories_plotted',runs={k:str(p)for k,p in runs.items()},
        last_steps={k:rows[-1]['step']for k,rows in histories.items()},common_step=common,
        common_metrics={k:next(r for r in rows if r['step']==common)for k,rows in histories.items()},
        histories=histories,config_differences_only_time_sampling_causal_and_name=True,
        scope='No smoothing/extrapolation/newpredictions; one reused developmentgeometry/initialization. Full final sampling/state proofs are separate. Pooled unweighted minority error is not the weighted terminal contour/shape metric.')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output/'convergence.png'),common_step=common,last_steps=report['last_steps'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('uniform-bdf','adaptive-bdf','uniform-jvp','adaptive-jvp','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();plot(dict(uniform_BDF=a.uniform_bdf,adaptive_BDF=a.adaptive_bdf,
        uniform_JVP=a.uniform_jvp,adaptive_JVP=a.adaptive_jvp),a.output)
