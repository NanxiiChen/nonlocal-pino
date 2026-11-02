"""Plot actual saved NS--CH fields/convergence without running a solver."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from examples.channel_nsch2d.configs.train_base import get_config
from examples.channel_nsch2d.problem import walls


def plot_fields(path,output,field_label=None,*,selected_fields=None,selected_times=None):
    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    config_path=Path(path).parent/'config.json';c=get_config()
    from ml_collections import ConfigDict
    if config_path.exists():c=ConfigDict(json.loads(config_path.read_text()))
    elif 'identity_json' in d:
        model=json.loads(str(d['identity_json']))
        c.geometry=ConfigDict(model['geometry']);c.physical=ConfigDict(model['physical'])
    else:raise ValueError('Field needs physical identity or adjacent saved config')
    nx=len(np.unique(d['points'][:,0]));ny=len(d['points'])//nx
    if nx*ny!=len(d['points']):raise ValueError('Expected mapped tensor diagnostic queries')
    q=d['points'].reshape(nx,ny,2);xq=q[:,0,0]
    bq,tq=walls(q[...,0],d['amplitudes'],c,xp=np)
    eq=(q[...,1]-bq)/(tq-bq)
    # Dense JAX replay may retain FP32 query coordinates, unlike the FP64 FEM
    # file. Check mapped-grid alignment at its actual coordinate precision.
    if d['points'].dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Mapped plotting coordinates must be float32 or float64')
    coordinate_tolerance=max(1e-12,8*np.finfo(d['points'].dtype).eps)
    np.testing.assert_allclose(eq,np.broadcast_to(eq[0],eq.shape),atol=coordinate_tolerance,rtol=0)
    # Display interpolation only; metrics always use the original saved queries.
    dx,dy=max(nx,256),max(ny,64)
    xx,ee=np.meshgrid(np.linspace(0,c.geometry.length,dx+1),np.linspace(0,1,dy+1),indexing='ij')
    bottom,top=walls(xx,d['amplitudes'],c,xp=np);yy=bottom+ee*(top-bottom)
    xc,ec=np.meshgrid((np.arange(dx)+.5)*c.geometry.length/dx,(np.arange(dy)+.5)/dy,indexing='ij')
    display_query=np.stack((np.clip(xc,xq[0],xq[-1]),np.clip(ec,eq[0,0],eq[0,-1])),axis=-1)
    def smooth(values):
        return RegularGridInterpolator((xq,eq[0]),values.reshape(nx,ny))(display_query)
    selected=np.rint(np.linspace(0,len(d['time'])-1,5)).astype(int)
    if selected_times is not None:
        requested=np.asarray(selected_times,dtype=np.float64)
        if requested.ndim!=1 or len(requested)==0 or not np.isfinite(requested).all() or np.any(np.diff(requested)<=0):
            raise ValueError('Requested gallery times must be finite, nonempty and strictly increasing')
        selected=[]
        for t in requested:
            matches=np.flatnonzero(np.isclose(d['time'],t,atol=1e-10,rtol=0))
            if len(matches)!=1:raise ValueError(f'Requested gallery time {t} is absent or ambiguous; no temporal interpolation')
            selected.append(int(matches[0]))
        selected=np.asarray(selected)
    specs=[('phi',0),('speed',None),('pressure',4)]
    if selected_fields is not None:
        if not selected_fields or len(set(selected_fields))!=len(selected_fields) or set(selected_fields)-{name for name,_ in specs}:
            raise ValueError('Gallery fields must be a nonempty unique selection of phi/speed/pressure')
        specs=[(name,index) for name,index in specs if name in selected_fields]
    scale_indices=slice(None) if selected_times is None else selected
    target=d['reference'] if 'reference' in d else d['fields'];predicted=d['fields']
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,index in specs:
        true=target[...,index] if index is not None else np.linalg.norm(target[...,2:4],axis=-1)
        pred=predicted[...,index] if index is not None else np.linalg.norm(predicted[...,2:4],axis=-1)
        rows=3 if 'reference' in d else 1
        row_labels=('FEM reference','Prediction','Absolute error') if rows==3 else (field_label or 'Field',)
        fig,axes=plt.subplots(rows,len(selected),figsize=(3.4*len(selected),2.1*rows),squeeze=False,layout='constrained')
        vmin,vmax=(-1.,1.) if name=='phi' else (min(float(true[scale_indices].min()),float(pred[scale_indices].min())),max(float(true[scale_indices].max()),float(pred[scale_indices].max())))
        if vmax==vmin:vmax=vmin+1.
        for row in range(rows):
            series=(true,pred,np.abs(pred-true))[row]
            high=vmax if row<2 else max(float(np.abs(pred[scale_indices]-true[scale_indices]).max()),1e-8)
            low=vmin if row<2 else 0.
            for col,i in enumerate(selected):
                ax=axes[row,col]
                mesh=ax.pcolormesh(xx.T,yy.T,smooth(series[i]).T,shading='flat',
                    cmap=('coolwarm' if name=='phi' else 'viridis') if row<2 else 'magma',vmin=low,vmax=high,rasterized=True)
                x=np.linspace(0,c.geometry.length,401);b,t=walls(x,d['amplitudes'],c,xp=np)
                ax.plot(x,b,color='.25',lw=.8);ax.plot(x,t,color='.25',lw=.8)
                ax.set(xlim=(0,c.geometry.length),ylim=(0,c.geometry.height),aspect='equal')
                ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(f't={d["time"][i]:g}')
                if col==0:ax.set_ylabel(row_labels[row])
            fig.colorbar(mesh,ax=axes[row,:].tolist(),shrink=.85,pad=.012)
        from examples.channel_nsch2d.geometry import random_walls
        label='independent random spline walls' if random_walls(c) else 'upper/lower indentation=('+', '.join(f'{a:g}' for a in d['amplitudes'])+')'
        fig.suptitle(f'Full NS–CH: {name}, {label}\n'
                     f'{nx}×{ny} saved queries; linear display interpolation')
        fig.savefig(output/f'{name}-fields.png',dpi=200);fig.savefig(output/f'{name}-fields.pdf');plt.close(fig)
    metrics=Path(path).parent/'metrics.jsonl'
    if metrics.exists():
        history=[json.loads(line) for line in metrics.read_text().splitlines()]
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for field in ('phi','velocity','pressure'):
            ax.semilogy([r['step'] for r in history],[100*r[field] for r in history],label=field)
        ax.set(xlabel='Training updates',ylabel='Relative L2 (%)');ax.legend();ax.grid(alpha=.2)
        fig.savefig(output/'convergence.png',dpi=200);fig.savefig(output/'convergence.pdf');plt.close(fig)
    print(str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--field-label',help='Explicit label for a single-row field plot')
    parser.add_argument('--field',action='append',choices=('phi','speed','pressure'),help='Plot only requested fields; repeat to select several')
    parser.add_argument('--times',nargs='+',type=float,help='Explicit increasing saved times; display scales use these snapshots')
    a=parser.parse_args();plot_fields(a.fields,a.output,a.field_label,selected_fields=a.field,selected_times=a.times)
