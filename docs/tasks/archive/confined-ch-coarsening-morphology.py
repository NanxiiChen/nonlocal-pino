"""Reference-only morphology comparison; distinct physical IC/epsilon/time cases."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

OUT=Path(__file__).parent
ROOT=Path('/root/autodl-tmp/data/confined_ch2d')


def read(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def morphology(d,field):
    cells=d['mesh_cells'];owners={};i=[];j=[]
    for k,tri in enumerate(cells):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edge=tuple(sorted((int(a),int(b))))
            if edge in owners:i.extend((k,owners[edge]));j.extend((owners[edge],k))
            else:owners[edge]=k
    adjacency=sparse.coo_matrix((np.ones(len(i)),(i,j)),shape=(len(cells),len(cells))).tocsr()
    v=d['volumes'];rows=[]
    for t,p in zip(d['time'],d[field],strict=True):
        row=dict(time=float(t),pure_fraction=float(v[abs(p)>.9].sum()/v.sum()),
            mixed_fraction=float(v[abs(p)<.8].sum()/v.sum()),minimum=float(p.min()),maximum=float(p.max()),phases={})
        for label,mask in [('positive',p>0),('negative',p<0)]:
            n,groups=connected_components(adjacency[mask][:,mask],directed=False)
            areas=np.sort(np.bincount(groups,weights=v[mask],minlength=n))[::-1]/v.sum()
            row['phases'][label]=dict(area_fractions=areas.tolist(),count=int(n),significant_count=int(np.sum(areas>=.005)))
        row['significant_components']=sum(a['significant_count'] for a in row['phases'].values())
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'coarsening/preflight.npz')
    parser.add_argument('--prefix',default='confined-ch-coarsening')
    parser.add_argument('--stage',choices=('preflight','validated-reference'),default='preflight')
    args=parser.parse_args()
    if args.stage=='validated-reference':
        checks=json.loads(args.candidate.with_suffix('.json').read_text())
        if checks['status']!='passed':raise ValueError('Cannot label a failed reference as validated')
    cases={'old':read(ROOT/'reference.npz'),'richer':read(args.candidate)}
    fields={'old':'phi_local','richer':'phi' if args.stage=='preflight' else 'phi_local'}
    stats={name:morphology(d,fields[name]) for name,d in cases.items()}
    role='coarse FEM preflight' if args.stage=='preflight' else 'refinement-validated local FEM reference'
    result=dict(scope=f'Different prescribed physical cases, not an accuracy ranking. New fields are {role}, not PINN predictions.',significant_area_fraction=.005,cases=stats)
    result['candidate_path']=str(args.candidate)
    (OUT/(args.prefix+'-morphology.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,5,figsize=(14,6.3),layout='constrained')
    for row,(name,d) in enumerate(cases.items()):
        mesh=mtri.Triangulation(d['mesh_points'][:,0],d['mesh_points'][:,1],d['mesh_cells'])
        select=np.linspace(0,len(d['time'])-1,5).round().astype(int)
        for col,index in enumerate(select):
            im=axes[row,col].tripcolor(mesh,facecolors=d[fields[name]][index],cmap='coolwarm',vmin=-1,vmax=1,rasterized=True)
            axes[row,col].set(aspect='equal',title=f't={d["time"][index]:.2f}');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if col==0:axes[row,col].set_ylabel('Old reference, epsilon=.05' if name=='old' else f'Richer {args.stage}, epsilon=.02')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='phi (same scale)')
    fig.savefig(OUT/(args.prefix+'-fields.png'),dpi=180);plt.close(fig)
    d=cases['richer'];mesh=mtri.Triangulation(d['mesh_points'][:,0],d['mesh_points'][:,1],d['mesh_cells'])
    fig,axes=plt.subplots(1,5,figsize=(14,3.2),layout='constrained')
    # Resolve the rapid early restructuring as well as the requested terminal time.
    indices=np.rint(np.array([0,.1,.3,.6,1])*(len(d['time'])-1)).astype(int)
    for ax,index in zip(axes,indices,strict=True):
        im=ax.tripcolor(mesh,facecolors=d[fields['richer']][index],cmap='coolwarm',vmin=-1,vmax=1,rasterized=True)
        ax.set(aspect='equal',title=f't*={d["time"][index]:.2f}');ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'Richer confined CH: {role}')
    fig.colorbar(im,ax=axes.tolist(),shrink=.8,label='phi')
    fig.savefig(OUT/(args.prefix+'-selected-fields.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for name,rows in stats.items():
        axes[0].plot([r['time'] for r in rows],[r['pure_fraction'] for r in rows],label=name)
        axes[1].step([r['time'] for r in rows],[r['significant_components'] for r in rows],where='post',label=name)
    axes[0].set(xlabel='Time',ylabel='Area fraction |phi| > 0.9')
    axes[1].set(xlabel='Time',ylabel='Significant positive + negative components')
    for ax in axes:ax.legend();ax.grid(alpha=.25)
    fig.savefig(OUT/(args.prefix+'-structure.png'),dpi=180);plt.close(fig)
    for name,rows in stats.items():
        print(name,json.dumps([{k:r[k] for k in ('time','pure_fraction','significant_components','minimum','maximum')} for r in (rows[0],rows[-1])]))


if __name__=='__main__':main()
