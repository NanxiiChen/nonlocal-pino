"""Measured L2/weight histories and matched-query drop contours for nine weights."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from docs.tasks.channel_droplet_metrics import contour_samples


def plot(control,trial,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    control,trial,output=map(Path,(control,trial,output))
    if output.exists():raise FileExistsError(output)
    gate=json.loads((trial/'balance-control-validation.json').read_text())
    if gate['status']!='matched_cold_state_all_inputs_time_and_nine_weight_history_passed':
        raise ValueError('Complete matched control audit required before plotting')
    read=lambda p:[json.loads(s)for s in p.read_text().splitlines()]
    histories=[read(p/'metrics.jsonl')for p in (control,trial)]
    if any(rows[0]['step']!=0 or rows[-1]['step']!=5000 for rows in histories):
        raise ValueError('Complete cold5000 measured histories required')
    labels=('Fixed residual scales','Raw residuals + nine adaptive weights')
    colors=('#0072B2','#D55E00')
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.2),layout='constrained')
    for ax,key,title in zip(axes.ravel(),('phi','terminal_phi','velocity','pressure'),
            ('Phase: all saved times','Phase: terminal','Velocity','Pressure'),strict=True):
        for rows,label,color in zip(histories,labels,colors,strict=True):
            ax.semilogy([r['step']for r in rows],100*np.array([r[key]for r in rows]),
                label=label,color=color,lw=1.8)
        ax.set(title=title,xlabel='SOAP updates',ylabel='Relative L2 (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Same cold model / SOAP / geometry and time draws / local FEM target')
    output.mkdir(parents=True);fig.savefig(output/'convergence.png',dpi=200)
    fig.savefig(output/'convergence.pdf');plt.close(fig)
    weights=read(trial/'balance.jsonl');names=weights[0]['groups']
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for i,name in enumerate(names):
        ax.semilogy([r['step']for r in weights],[r['weights'][i]for r in weights],label=name)
    ax.set(xlabel='SOAP update (refresh steps)',ylabel='Adaptive weight',title='Nine independent stopped weights')
    ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8,frameon=False)
    fig.savefig(output/'weights.png',dpi=200);plt.close(fig)
    fields=[]
    for p in (control,trial):
        with np.load(p/'prediction.npz',allow_pickle=False)as f:fields.append({k:f[k]for k in f.files})
    for key in ('reference','points','time','amplitudes'):
        np.testing.assert_array_equal(fields[0][key],fields[1][key])
    d=fields[0];points=d['points'];nx=len(np.unique(points[:,0]));shape=(nx,len(points)//nx)
    selected=[int(np.flatnonzero(d['time']==t)[0])for t in (0.,1.5,3.,4.,6.)]
    fig,axes=plt.subplots(1,5,figsize=(14,3.3),layout='constrained')
    for ax,i in zip(axes,selected,strict=True):
        visible=[]
        series=((d['reference'],'black','Local FEM','-'),
            (d['fields'],colors[0],labels[0],'-'),(fields[1]['fields'],colors[1],labels[1],'--'))
        for values,color,label,style in series:
            _,_,curves=contour_samples(points,values[i,:,0],shape,1024)
            visible.extend(curves)
            if not curves:
                ax.text(.02,.02,f'{label}: no zero contour',transform=ax.transAxes,color=color,fontsize=6)
            for j,curve in enumerate(curves):ax.plot(*curve.T,color=color,ls=style,lw=1.6,label=label if j==0 else None)
        cloud,measure,_=contour_samples(points,d['reference'][i,:,0],shape,1024)
        if not len(cloud):raise ValueError('Expected local FEM droplet contour')
        center=np.sum(cloud*measure[:,None],axis=0)/measure.sum();half=.35
        extent=np.concatenate(visible)
        lo=np.minimum(extent.min(0)-.04,center-half);hi=np.maximum(extent.max(0)+.04,center+half)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),
            aspect='equal',xlabel='x',title=f't={d["time"][i]:g}');ax.grid(alpha=.15)
    axes[0].set_ylabel('y');axes[0].legend(fontsize=6.5,frameon=False)
    fig.suptitle('Identical physical queries; t=4 peak flattening included')
    fig.savefig(output/'contours.png',dpi=200);fig.savefig(output/'contours.pdf');plt.close(fig)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    result=dict(status='actual_l2_weights_and_matched_contours_plotted',control=str(control),trial=str(trial),
        actual_histories_only=True,same_target_points_times=True,contour_times=d['time'][selected].tolist(),
        scope='One nominal development geometry/initialization; no smoothed or extrapolated training metrics')
    (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','trial','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(plot(a.control,a.trial,a.output)),flush=True)
