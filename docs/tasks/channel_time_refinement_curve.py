"""Plot the actual shared-parent temporal-refinement development histories."""
from pathlib import Path
import argparse
import json


def plot(original,refined,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=Path(original);refined=Path(refined);output=Path(output)
    configs=[json.loads((p/'config.json').read_text()) for p in (original,refined)]
    if configs[0]['train']['continuation']['parent_run']!=configs[1]['train']['continuation']['parent_run']:
        raise ValueError('Comparison requires the same complete model/optimizer parent')
    rows=[]
    for p in (original,refined):
        history=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()]
        if any(b['step']<=a['step'] for a,b in zip(history,history[1:])):
            raise ValueError('Metric history is not strictly ordered')
        rows.append([r for r in history if 10000<=r['step']<=15000])
    if any(not history for history in rows):raise ValueError('Both measured histories are required')
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for ax,key,title in zip(axes.ravel(),('phi','terminal_phi','velocity','pressure'),
                           ('Phase: all snapshots','Phase: terminal time','Velocity: positive times','Pressure: positive times'),strict=True):
        for data,config,color in zip(rows,configs,('#777777','#0072B2'),strict=True):
            x=[r['step'] for r in data];y=[100*r[key] for r in data]
            ax.plot(x,y,color=color,lw=1.8,label=f"BDF2 dt={config['time_discretization']['dt']}")
            ax.scatter(x[-1],y[-1],color=color,s=18,zorder=3)
        ax.axhline(10,color='#B22222',ls=':',lw=1)
        ax.set(title=title,xlabel='Global optimizer updates',ylabel='Relative L2 (%)',xlim=(10000,15000))
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Same complete 10k parent; one development geometry, 384 FEM query points',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=200);fig.savefig(output.with_suffix('.pdf'));plt.close(fig)
    receipt=dict(original_run=str(original),refined_run=str(refined),
        last_steps=[r[-1]['step'] for r in rows],
        scope='Actual one-seed/development histories; no extrapolation, unseen-geometry or dense-field claim',
        histories=rows)
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output.with_suffix('.png')),last_steps=receipt['last_steps'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',required=True)
    p.add_argument('--refined',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plot(a.original,a.refined,a.output)
