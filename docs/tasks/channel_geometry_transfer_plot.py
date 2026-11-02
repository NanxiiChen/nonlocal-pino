"""Matched-query frozen and newly trained geometry contours, at explicit times."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from ml_collections import ConfigDict
from docs.tasks.channel_droplet_metrics import contour_samples
from examples.channel_nsch2d.problem import walls


def plot(frozen,trained,output,times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen,trained,output=map(Path,(frozen,trained,output))
    if output.exists():raise FileExistsError(output)
    fields=[]
    for p in (frozen,trained):
        with np.load(p,allow_pickle=False)as f:fields.append({k:f[k]for k in f.files})
    for key in ('reference','points','time','amplitudes'):
        np.testing.assert_array_equal(fields[0][key],fields[1][key])
    c=ConfigDict(json.loads((trained.parent/'config.json').read_text()))
    if not times or np.any(np.diff(times)<=0):raise ValueError('Explicit increasing saved times required')
    selected=[]
    for t in times:
        ids=np.flatnonzero(np.isclose(fields[1]['time'],t,atol=1e-10,rtol=0))
        if len(ids)!=1:raise ValueError(f'Requested time{t} absent or ambiguous')
        selected.append(int(ids[0]))
    points=fields[1]['points'];nx=len(np.unique(points[:,0]));shape=(nx,len(points)//nx)
    fig,axes=plt.subplots(1,len(times),figsize=(2.9*len(times),3.2),squeeze=False,layout='constrained')
    for ax,i in zip(axes[0],selected,strict=True):
        values=((fields[1]['reference'],'black','Local FEM','-'),
            (fields[0]['fields'],'#D55E00','Frozen extrapolation','--'),
            (fields[1]['fields'],'#0072B2','New-family cold5000','-'))
        for series,color,label,style in values:
            _,_,curves=contour_samples(points,series[i,:,0],shape,1024)
            for j,curve in enumerate(curves):ax.plot(*curve.T,color=color,ls=style,lw=1.7,label=label if j==0 else None)
        cloud,measure,_=contour_samples(points,fields[1]['reference'][i,:,0],shape,1024)
        if not len(cloud):raise ValueError('Reference contour absent')
        center=np.sum(cloud*measure[:,None],axis=0)/measure.sum();half=1.6*c.initial.radius
        x=np.linspace(center[0]-half,center[0]+half,201);bottom,top=walls(x,fields[1]['amplitudes'],c,xp=np)
        ax.plot(x,bottom,color='.65',lw=.8);ax.plot(x,top,color='.65',lw=.8)
        ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
            aspect='equal',title=f't={fields[1]["time"][i]:g}',xlabel='x')
        ax.grid(alpha=.15)
    axes[0,0].set_ylabel('y');axes[0,0].legend(fontsize=7,frameon=False)
    fig.suptitle('Stronger constriction: identical FEM target and physical queries; peak flattening at t=4 included')
    output.mkdir(parents=True);fig.savefig(output/'contours.png',dpi=200)
    fig.savefig(output/'contours.pdf');plt.close(fig)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    (output/'validation.json').write_text(json.dumps(dict(status='common_query_geometry_contours_plotted',
        frozen=str(frozen),trained=str(trained),times=times,reference_points_times_geometry_exact=True,
        scope='Saved-field display only; no inference, training, labels or normalization changes'),indent=2)+'\n')
    print(str(output/'contours.png'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','trained','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--times',required=True,nargs='+',type=float)
    a=p.parse_args();plot(a.frozen,a.trained,a.output,a.times)
