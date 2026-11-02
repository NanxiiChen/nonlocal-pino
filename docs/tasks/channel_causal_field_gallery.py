"""CPU-only saved-field panels and matched causal/time-JVP contour zooms."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples.channel_nsch2d.evaluate import plot_fields
from docs.tasks.channel_droplet_metrics import contour_samples


def read_run(path):
    if not path.is_absolute():raise ValueError('Gallery run paths must be absolute')
    summary=json.loads((path/'summary.json').read_text())
    config=json.loads((path/'config.json').read_text())
    if summary['status']!='completed' or summary['metrics']['step']!=summary['iterations']:
        raise ValueError('Gallery requires the prescribed completed endpoint')
    with np.load(path/'prediction.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    fields,reference=data['fields'],data['reference']
    if (fields.shape!=reference.shape or fields.ndim!=3 or fields.shape[-1]!=5
            or fields.shape[:2]!=(len(data['time']),len(data['points']))
            or data['points'].shape!=(fields.shape[1],2)
            or not all(np.isfinite(v).all() for v in data.values())):
        raise ValueError('Finite aligned fivefield arrays/points/times are required')
    if np.any(np.diff(data['time'])<=0):raise ValueError('Saved times must be strictly increasing')
    positive=data['time']>0
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics=dict(phi=relative(fields[...,0],reference[...,0]),
        terminal_phi=relative(fields[-1,:,0],reference[-1,:,0]),
        velocity=relative(fields[positive,:,2:4],reference[positive,:,2:4]),
        pressure=relative(fields[positive,:,4],reference[positive,:,4]))
    gap=max(abs(metrics[k]-summary['metrics'][k]) for k in metrics)
    if gap>2e-6:raise ValueError(f'Saved field metrics disagree with endpoint: {gap}')
    return data,summary,config,dict(run=str(path),metrics=metrics,metric_maximum_gap=gap)


def main(control,trial,output):
    control,trial,output=map(Path,(control,trial,output))
    if not output.is_absolute():raise ValueError('Gallery output must be absolute')
    if output.exists():raise FileExistsError(output)
    a,sa,ca,ra=read_run(control);b,sb,cb,rb=read_run(trial)
    if sa['iterations']!=sb['iterations']:raise ValueError('Require matching update budgets')
    for key in ('reference','points','time','amplitudes'):
        np.testing.assert_array_equal(a[key],b[key],err_msg=f'Gallery {key} differs')
    for key in ('geometry','physical','initial'):
        if ca[key]!=cb[key]:raise ValueError(f'Gallery physical identity differs: {key}')
    if 'adaptation' not in cb['causal']:raise ValueError('Trial must explicitly select adaptive causal')
    times=(0.,1.5,3.,4.5,6.);selected=[]
    for t in times:
        ids=np.flatnonzero(np.isclose(b['time'],t,rtol=0,atol=1e-10))
        if len(ids)!=1:raise ValueError(f'Exactly saved snapshot {t} required; no time interpolation')
        selected.append(int(ids[0]))
    points=b['points'];nx=len(np.unique(points[:,0]));ny=len(points)//nx
    if nx*ny!=len(points):raise ValueError('Mapped tensor queries required')
    series=(b['reference'],a['fields'],b['fields'])
    def label(config):
        if 'stream_lifting' in config:return 'Normalized inlet JVP PINO'
        if 'time_derivative' in config:return 'Time JVP PINO'
        if 'causal' in config:return 'Adaptive BDF PINO' if 'adaptation' in config['causal'] else 'Fixed causal BDF PINO'
        return 'Uniform BDF PINO'
    labels=('Local FEM',label(ca),label(cb))
    colors=('black','#0072B2','#D55E00');styles=('-','--','-')
    panels=[];centers=[];half=1.6*cb['initial']['radius'];counts=[]
    for i in selected:
        cloud,measure,_=contour_samples(points,b['reference'][i,:,0],(nx,ny),1024)
        if not len(cloud):raise ValueError('Reference zero contour absent')
        center=np.sum(cloud*measure[:,None],axis=0)/measure.sum()
        contours=[contour_samples(points,s[i,:,0],(nx,ny),1024)[2] for s in series]
        for curves in contours:
            for curve in curves:half=max(half,1.08*float(np.max(np.abs(curve-center))))
        centers.append(center);panels.append(contours);counts.append([len(v) for v in contours])
    output.mkdir(parents=True)
    plot_fields(trial/'prediction.npz',output,selected_fields=('phi',),selected_times=times)
    flow_times=(.5,1.5,3.,4.5,6.)
    # The FEM snapshot stores the unsolved initial pressure as zero. Use
    # positive-time flow panels, matching the velocity/pressure L2 definition.
    plot_fields(trial/'prediction.npz',output,selected_fields=('speed','pressure'),selected_times=flow_times)
    fig,axes=plt.subplots(1,len(selected),figsize=(15,3.8),layout='constrained')
    for ax,i,center,contours in zip(axes,selected,centers,panels,strict=True):
        for curves,color,style,label in zip(contours,colors,styles,labels,strict=True):
            for curve in curves:ax.plot(*curve.T,color=color,ls=style,lw=1.8)
            if not curves:ax.text(.03,.05,f'{label}: no zero contour',transform=ax.transAxes,color=color,fontsize=7)
        ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
            aspect='equal',title=f't = {b["time"][i]:g}',xlabel='x')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('y')
    handles=[Line2D([],[],color=c,ls=s,lw=1.8,label=l)for c,s,l in zip(colors,styles,labels,strict=True)]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(f'Zero-phase contours, same physical zoom scale | {sa["iterations"]} updates each')
    fig.savefig(output/'contour-comparison.png',dpi=220)
    fig.savefig(output/'contour-comparison.pdf');plt.close(fig)
    result=dict(status='saved_fields_common_query_gallery_validated',control=ra,adaptive=rb,
        updates=sa['iterations'],times=list(times),saved_query_grid=[nx,ny],
        field_times=dict(phi=list(times),speed=list(flow_times),pressure=list(flow_times)),
        flow_initial_pressure_note='FEM t0pressure is its unsolved zero initialization; positive-time flow panels match the logged flow L2 scope',
        same_reference_points_times_geometry=True,contour_counts=counts,
        contour_series=list(labels),zoom_half_width=half,zoom_centers=np.asarray(centers).tolist(),
        images={k:str(output/(k+'.png'))for k in ('phi-fields','speed-fields','pressure-fields','contour-comparison')},
        scope='One nominal development geometry; original saved arrays for metrics/contours, spatial interpolation only for field display, no time interpolation or reference solve')
    (output/'gallery-validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',required=True);p.add_argument('--trial',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.control,a.trial,a.output)
