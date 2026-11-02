"""Compare saved GMLS/integral errors and droplet contours; no model execution."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot(control,integral,output):
    control,integral,output=Path(control),Path(integral),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    colors=('#3266a8','#d56626')
    paths=(control,integral);labels=('GMLS','Moment-corrected integral')
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.3),constrained_layout=True)
    for path,label,color in zip(paths,labels,colors,strict=True):
        rows=[json.loads(s)for s in (path/'metrics.jsonl').read_text().splitlines()]
        steps=[r['step']for r in rows]
        for ax,key,title in zip(axes,('phi','velocity','pressure'),('Phase field','Velocity','Pressure'),strict=True):
            ax.semilogy(steps,[100*r[key]for r in rows],label=label,color=color,lw=1.7)
            ax.set(title=title,xlabel='Optimizer updates',ylabel='Relative L2 error (%)')
            ax.grid(alpha=.2,which='both')
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(output/'convergence.png',dpi=180);plt.close(fig)
    with np.load(control/'prediction.npz',allow_pickle=False)as f:
        x=f['points'];time=f['time'];reference=f['reference'];gmls=f['fields']
    with np.load(integral/'prediction.npz',allow_pickle=False)as f:
        for key,expected in (('points',x),('time',time),('reference',reference)):
            np.testing.assert_array_equal(f[key],expected)
        predicted=f['fields']
    selected=[0.,1.5,3.,4.,6.]
    tri=mtri.Triangulation(x[:,0],x[:,1])
    fig,axes=plt.subplots(1,5,figsize=(14,3),constrained_layout=True)
    from examples.channel_nsch2d.problem import walls
    from ml_collections import ConfigDict
    c=ConfigDict(json.loads((integral/'config.json').read_text()))
    wall_x=np.linspace(0,c.geometry.length,1025)
    bottom,top=walls(wall_x,(c.geometry.upper,c.geometry.lower),c,xp=np)
    for ax,t in zip(axes,selected,strict=True):
        ids=np.flatnonzero(np.isclose(time,t,atol=1e-12,rtol=0))
        if len(ids)!=1:raise ValueError(f'Saved physical time absent or ambiguous: {t}')
        i=ids[0]
        for field,color,style in ((reference,'black','-'),(gmls,colors[0],'--'),(predicted,colors[1],'-')):
            ax.tricontour(tri,field[i,:,0],levels=[0.],colors=[color],linestyles=[style],linewidths=1.6)
        ax.plot(wall_x,bottom,color='.5',lw=.8);ax.plot(wall_x,top,color='.5',lw=.8)
        occupied=np.any(np.stack([field[i,:,0]<0 for field in (reference,gmls,predicted)]),axis=0)
        if not np.any(occupied):raise ValueError(f'No dispersed phase to frame at time{t}')
        left,right=x[occupied,0].min(),x[occupied,0].max()
        center=(left+right)/2;half=max(.75,(right-left)/2+.2)
        ax.set(xlim=(max(0,center-half),min(c.geometry.length,center+half)),ylim=(-.03,1.03),title=f't = {t:g}',xlabel='x')
        ax.set_aspect('equal')
    axes[0].set_ylabel('y')
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([],[],color='black',label='Local FEM'),
        Line2D([],[],color=colors[0],ls='--',label=labels[0]),
        Line2D([],[],color=colors[1],label='Integral')],frameon=False,fontsize=7,loc='lower left')
    fig.savefig(output/'contours.png',dpi=200);plt.close(fig)
    (output/'provenance.json').write_text(json.dumps(dict(control=str(control),integral=str(integral),
        times=selected,reference='identical saved local FEM',interface='phi=0',
        scope='One nominal development geometry, no population estimate'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','integral','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();plot(a.control,a.integral,a.output)
