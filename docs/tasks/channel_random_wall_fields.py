"""Four qualified random-wall trajectories: physical fields and aligned shapes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import contourpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from scipy.spatial import cKDTree
from examples.channel_nsch2d.problem import walls
from examples.channel_nsch2d.geometry import geometry_statistics,nominal_parameters
from docs.tasks.channel_droplet_metrics import contour_samples


def main(root,output):
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False)
    c=ConfigDict(json.loads((root/'config.json').read_text()))
    cohort=json.loads((root/'unseen3'/'manifest.json').read_text())
    rows=[dict(root=str(root/'development'),amplitudes=nominal_parameters(c))]+cohort['cases']
    datasets=[];histories=[];selected=[0,3,6,9,12]
    colors=['#222222','#D55E00','#0072B2','#009E73']
    fig,axes=plt.subplots(4,5,figsize=(16,7.4),layout='constrained')
    for k,row in enumerate(rows):
        folder=Path(row['root']);gate=json.loads((folder/'reference-validation.json').read_text())
        if gate['status']!='passed':raise ValueError('Do not plot an unqualified reference as accepted')
        with np.load(gate['reference']) as f:d={key:f[key] for key in f.files}
        q=d['points'].reshape(96,40,2);b,t=walls(q[...,0],d['amplitudes'],c,xp=np)
        measure=(t-b).ravel()*4/(96*40);history=[]
        for j,time in enumerate(d['time']):
            phi=d['fields'][j,:,0];minority=np.clip((1-phi)/2,0,1);mass=minority@measure
            center=np.sum(q.reshape(-1,2)*(minority*measure)[:,None],axis=0)/mass
            dx=q.reshape(-1,2)-center
            cov=np.einsum('n,ni,nj->ij',measure*minority,dx,dx)/mass
            eig=np.linalg.eigvalsh(cov)
            cloud,weights,curves=contour_samples(d['points'],phi,(96,40),1024)
            if len(curves)!=1:raise ValueError('Connected-droplet pilot needs one resolved contour')
            p=d['fields'][j,:,4].reshape(96,40)
            history.append(dict(time=float(time),centroid=center.tolist(),moment_aspect=float(np.sqrt(eig[1]/eig[0])),
                contour_x_over_y_span=float(np.ptp(cloud[:,0])/np.ptp(cloud[:,1])),
                pressure_drop=float(np.mean(p[0])-np.mean(p[-1])),negative_area=float(measure@(phi<0)),
                centered_contour=cloud-center,curve=curves[0]))
        datasets.append(d);histories.append(history)
        xx=np.linspace(0,4,801);lower,upper=walls(xx,d['amplitudes'],c,xp=np)
        for col,j in enumerate(selected):
            ax=axes[k,col];ph=d['fields'][j,:,0].reshape(96,40)
            ax.contourf(q[...,0].T,q[...,1].T,ph.T,levels=np.linspace(-1,1,41),cmap='coolwarm',extend='both')
            curve=history[j]['curve'];ax.plot(curve[:,0],curve[:,1],color='.15',lw=.8)
            ax.fill_between(xx,0,lower,color='.65');ax.fill_between(xx,upper,1,color='.65')
            ax.plot(xx,lower,color='.2',lw=.8);ax.plot(xx,upper,color='.2',lw=.8)
            ax.set(xlim=(0,4),ylim=(0,1),aspect='equal',xticks=[],yticks=[])
            if col==0:ax.set_ylabel('Nominal'if k==0 else f'Unseen {k}')
            if k==0:ax.set_title(f't={d["time"][j]:g}')
    fig.suptitle('Independent FEM references: identical initial droplet, flux and material; random stationary walls')
    fig.savefig(output/'reference-fields.png',dpi=190);fig.savefig(output/'reference-fields.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(14,3.2),layout='constrained')
    pairwise=[]
    for ax,j in zip(axes,selected):
        for k,h in enumerate(histories):
            curve=h[j]['curve']-np.asarray(h[j]['centroid'])
            ax.plot(curve[:,0],curve[:,1],color=colors[k],label='Nominal'if k==0 else f'Unseen {k}')
        distances=[]
        for k in range(4):
            for m in range(k):
                a=histories[k][j]['centered_contour'];b=histories[m][j]['centered_contour']
                distances.append(float((cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean())/2/c.initial.radius))
        pairwise.append(dict(time=float(datasets[0]['time'][j]),mean_aligned_distance_over_R0=float(np.mean(distances)),maximum_pairwise_distance_over_R0=max(distances)))
        ax.set(xlim=(-.37,.37),ylim=(-.37,.37),aspect='equal',title=f't={datasets[0]["time"][j]:g}',xlabel='x - centroid')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('y - centroid');axes[-1].legend(fontsize=8)
    fig.suptitle('Centroid-aligned zero contours: separates deformation from displacement')
    fig.savefig(output/'aligned-shapes.png',dpi=190);fig.savefig(output/'aligned-shapes.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,3.2),layout='constrained')
    for k,h in enumerate(histories):
        ts=[v['time']for v in h];label='Nominal'if k==0 else f'Unseen {k}'
        for ax,values in zip(axes,([v['centroid'][0]for v in h],[v['centroid'][1]for v in h],
                                 [v['moment_aspect']for v in h],[v['pressure_drop']for v in h])):
            ax.plot(ts,values,color=colors[k],label=label)
    for ax,title in zip(axes,('Centroid x','Centroid y','Minority second-moment aspect','Pressure difference at end query sections')):
        ax.set(xlabel='Time',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(output/'response-histories.png',dpi=190);fig.savefig(output/'response-histories.pdf');plt.close(fig)
    report=dict(status='four_refined_reference_trajectories_inspected',root=str(root),
        geometry_statistics=[geometry_statistics(d['amplitudes'],c)for d in datasets],
        centroid_aligned_pairwise=pairwise,
        histories=[[{k:v for k,v in r.items()if k not in ('centered_contour','curve')}for r in h]for h in histories],
        scope='One nominal and three predeclared independent prior draws; numerical response diversity, no neural prediction or population claim')
    (output/'response.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'plot-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(pairwise,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
