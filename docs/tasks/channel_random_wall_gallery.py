"""Postprocess the completed random-wall run; no model execution or training edits."""
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
from ml_collections import ConfigDict
from examples.channel_nsch2d.evaluate import plot_fields
from examples.channel_nsch2d.problem import walls,identity
from docs.tasks.channel_droplet_metrics import contour_samples,contour_distance


def gallery(run,output):
    run,output=Path(run),Path(output)
    summary=json.loads((run/'summary.json').read_text())
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if summary['status']!='completed' or summary['iterations']!=c.train.iterations:
        raise ValueError('Require a complete frozen run, not initial or partial weights')
    gate=json.loads((Path(c.reference.root)/'reference-validation.json').read_text())
    if gate['status']!='passed' or gate['identity']!=json.loads(json.dumps(identity(c))):
        raise ValueError('Actual saved geometry must match the independently refined FEM')
    with np.load(run/'prediction.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(gate['reference'],allow_pickle=False) as f:
        for key,expected in (('fields',d['reference']),('points',d['points']),('time',d['time']),('amplitudes',d['amplitudes'])):
            np.testing.assert_array_equal(f[key],expected)
    if d['fields'].shape!=d['reference'].shape or not np.isfinite(d['fields']).all():
        raise ValueError('Finite physical fields matching all FEM snapshots/queries required')
    v,r=d['fields'],d['reference']
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics=dict(phi=relative(v[...,0],r[...,0]),terminal_phi=relative(v[-1,:,0],r[-1,:,0]),
        velocity=relative(v[1:,:,2:4],r[1:,:,2:4]),pressure=relative(v[1:,:,4],r[1:,:,4]),
        chemical=relative(v[1:,:,1],r[1:,:,1]),dispersed_phase=relative((1-v[...,0])/2,(1-r[...,0])/2))
    for key,value in metrics.items():np.testing.assert_allclose(value,summary['metrics'][key],atol=2e-7,rtol=0)
    output.mkdir(parents=True,exist_ok=False)
    selected=[0.,1.5,3.,4.5,6.]
    plot_fields(run/'prediction.npz',output,selected_fields=['phi','speed'],selected_times=selected)
    # Initial pressure is not prescribed by the problem, so pressure begins at
    # the first positive saved time. This changes plotting only, not metrics.
    pressure_times=[.5,1.5,3.,4.5,6.]
    plot_fields(run/'prediction.npz',output,selected_fields=['pressure'],selected_times=pressure_times)
    nx=len(np.unique(d['points'][:,0]));ny=len(d['points'])//nx
    points=d['points'];lower,upper=walls(points[:,0],d['amplitudes'],c,xp=np)
    area=(upper-lower)*c.geometry.length/(nx*ny)
    fig,axes=plt.subplots(1,5,figsize=(14,3.4),layout='constrained');rows=[]
    wall_x=np.linspace(0,c.geometry.length,1601);bottom,top=walls(wall_x,d['amplitudes'],c,xp=np)
    for ax,t in zip(axes,selected,strict=True):
        ids=np.flatnonzero(np.isclose(d['time'],t,atol=1e-12,rtol=0))
        if len(ids)!=1:raise ValueError('No temporal interpolation or ambiguous snapshot selection')
        i=int(ids[0]);mask=r[i,:,0]<0
        if not np.any(mask):raise ValueError('Reference droplet absent at selected time')
        center=np.sum(points*(area*mask)[:,None],axis=0)/np.sum(area*mask)
        for field,color,style in ((r,'#222222','-'),(v,'#D55E00','--')):
            _,_,curves=contour_samples(points,field[i,:,0],(nx,ny),1024)
            for curve in curves:ax.plot(curve[:,0],curve[:,1],color=color,ls=style,lw=1.8)
        ax.fill_between(wall_x,0,bottom,color='.88');ax.fill_between(wall_x,top,1,color='.88')
        ax.plot(wall_x,bottom,color='.55',lw=.8);ax.plot(wall_x,top,color='.55',lw=.8)
        half=.40
        ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
            aspect='equal',title=f't = {t:g}',xlabel='x')
        ax.tick_params(labelsize=8);ax.grid(alpha=.18)
        p0=v[i,:,0]<0;r0=r[i,:,0]<0;union=area@(p0|r0)
        if union<=0:raise ValueError('Contour IoU denominator must be positive')
        distance=contour_distance(points,v[i,:,0],r[i,:,0],(nx,ny),1024)
        rows.append(dict(time=t,negative_phase_IoU=float((area@(p0&r0))/union),
            zero_contour=distance,minority_fraction_relative_L2=relative((1-v[i,:,0])/2,(1-r[i,:,0])/2)))
    axes[0].set_ylabel('y');axes[0].legend(handles=[Line2D([],[],color='#222222',label='Local FEM'),
        Line2D([],[],color='#D55E00',ls='--',label=f'PINO, {summary["iterations"]} updates')],fontsize=8,loc='lower left')
    fig.suptitle('Droplet zero contours at equal physical zoom; positions are not aligned or corrected')
    fig.savefig(output/'droplet-contours.png',dpi=200);fig.savefig(output/'droplet-contours.pdf');plt.close(fig)
    report=dict(status='actual_final_saved_fields_plotted',run=str(run),checkpoint=str(run/'model_final.eqx'),
        updates=summary['iterations'],reference=gate['reference'],query_shape=[nx,ny],
        phase_speed_times=selected,pressure_times=pressure_times,metrics=metrics,contours=rows,
        predicted_phi_range=[float(v[...,0].min()),float(v[...,0].max())],
        display='Mapped bilinear interpolation only; no temporal interpolation, field smoothing, clipping in metrics or contour/centroid correction',
        scope='Nominal development geometry; independent held-out evaluation remains a separate result')
    (output/'gallery-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'plot-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();gallery(a.run,a.output)
