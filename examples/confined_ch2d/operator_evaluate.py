"""Independent saved-field metrics and physical-mesh figures for the geometry NO."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from examples.confined_ch2d.operator_plotting import SmoothMeshDisplay
from examples.confined_ch2d.operator_problem import evaluation_indices
from examples.confined_ch2d.operator_history import training_records


def relative_error(a,b,weights):
    error=np.sum((a-b)**2*weights)
    norm=np.sum(b*b*weights)
    if norm==0:
        raise ValueError('Zero reference norm; report an absolute error instead')
    return float(np.sqrt(error/norm))


def check_chemical_relation(phi,predicted,reference,c):
    """Independent FP64 reconstruction with a declared FP32 rounding bound."""
    phi=np.asarray(phi,dtype=np.float64)
    coefficients=reference['coefficients'];neighbors=reference['neighbors']
    differences=phi[:,:,None]-phi[:,neighbors]
    action=np.sum(coefficients[None,:,:]*differences,axis=-1)
    p=c['physical'];mu=p['beta']*(phi**3-phi)+p['epsilon']**2*action
    dtype=np.dtype(c['train']['compute_dtype'])
    if dtype==np.dtype('float64'):
        np.testing.assert_allclose(mu,predicted,atol=1e-11,rtol=1e-12)
        return mu,action
    if dtype!=np.dtype('float32'):raise ValueError('Unsupported saved compute dtype')
    # Bound coefficient rounding, local arithmetic and a degree-term reduction.
    eps=np.finfo(dtype).eps;operations=neighbors.shape[1]+16
    gamma=operations*eps/(1-operations*eps)
    scale=abs(p['beta'])*(abs(phi)**3+abs(phi))
    scale+=p['epsilon']**2*np.sum(abs(coefficients[None,:,:]*differences),axis=-1)
    bound=gamma*(scale+np.finfo(dtype).tiny)
    difference=abs(np.asarray(predicted,dtype=np.float64)-mu)
    if not np.isfinite(predicted).all() or np.any(difference>bound):
        raise AssertionError(f'Saved FP32 chemical relation exceeds roundoff bound: max gap={difference.max()}')
    return np.asarray(predicted,dtype=np.float64),action


def evaluate(run,output,*,plot_style):
    if plot_style not in ('flat','smooth'):raise ValueError('Plot style must be flat or smooth')
    output.mkdir(parents=True,exist_ok=False)
    c=json.loads((run/'config.json').read_text());root=Path(c['operator']['dataset_path'])
    manifest=json.loads((root/'manifest.json').read_text())
    monitor,labeled=evaluation_indices(manifest)
    items=[];predictions={};references={}
    for record in manifest['conditions']:
        index=record['index']
        if index not in labeled:continue
        with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
            prediction={key:f[key] for key in f.files}
        with np.load(root/record['file'],allow_pickle=False) as f:
            ref={key:f[key] for key in f.files}
        for key in ('points','volumes','holes','time','mesh_points','mesh_cells'):
            np.testing.assert_array_equal(prediction[key],ref[key])
        x,v=ref['points'],ref['volumes'];phi=np.asarray(prediction['phi'],dtype=np.float64)
        dtype=np.dtype(c['train']['compute_dtype'])
        if dtype==np.dtype('float32'):
            np.testing.assert_array_equal(phi[0],ref['phi'][0].astype(dtype))
        else:np.testing.assert_allclose(phi[0],ref['phi'][0],atol=1e-13,rtol=0)
        mu,action=check_chemical_relation(phi,prediction['mu'],ref,c)
        energy=np.sum(v*(.25*c['physical']['beta']*(phi*phi-1)**2+.5*c['physical']['epsilon']**2*phi*action),axis=1)
        row=dict(index=index,phi_pooled=relative_error(phi,ref['phi'],v),
            phi_terminal=relative_error(phi[-1],ref['phi'][-1],v),
            mu_pooled=relative_error(mu,ref['mu'],v),mu_terminal=relative_error(mu[-1],ref['mu'][-1],v),
            frozen_phi_pooled=relative_error(np.broadcast_to(ref['phi'][0],phi.shape),ref['phi'],v),
            mass_max_drift=float(np.max(abs(phi@v/v.sum()-ref['phi'][0]@v/v.sum()))),
            energy_max_increase=float(max(0.,np.max(np.diff(energy)))))
        local=root/f'local_{index:03d}.npz'
        if local.exists():
            with np.load(local,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['points'],x)
                row['local_phi_pooled']=relative_error(phi,f['phi'],v)
                row['local_phi_terminal']=relative_error(phi[-1],f['phi'][-1],v)
                row['local_positive_mu_pooled']=relative_error(mu[1:],f['mu'][1:],v)
        items.append(row);predictions[index]=prediction;references[index]=ref
    report=dict(run=str(run),conditions=items,reference='Independent native graph/BDF',
        display=dict(style=plot_style,metric_scope='Unmodified physical centroid fields and cell areas',
                     smoothing='Area-weight incident cells to vertices, then linear interpolation inside original triangles; mask holes' if plot_style=='smooth' else 'None; flat original cell colors'))
    by_index={row['index']:row for row in items}
    report['train_monitor_indices']=monitor
    report['train_metric_scope']=f'{len(monitor)} labeled training monitors; no future reference for remaining training geometries'
    for name,ids in (('train',monitor),('test',manifest['test_indices'])):
        rows=[by_index[i] for i in ids]
        report[name]={key:float(np.mean([r[key] for r in rows])) for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal','frozen_phi_pooled')}
        report[name]['worst_phi_pooled']=max(r['phi_pooled'] for r in rows)
        report[name]['mass_max_drift']=max(r['mass_max_drift'] for r in rows)
        report[name]['energy_max_increase']=max(r['energy_max_increase'] for r in rows)
    local_rows=[row for row in items if 'local_phi_pooled' in row]
    report['reference']+=f'; separately refined local FEM on {len(local_rows)} held-out domains' if local_rows else '; local FEM not evaluated'
    report['local_FEM']=({key:float(np.mean([row[key] for row in local_rows])) for key in ('local_phi_pooled','local_phi_terminal','local_positive_mu_pooled')}
        if local_rows else dict(status='not_evaluated',conditions=0,reason='No local FEM reference files in this bank'))
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10})
    ids=manifest['test_indices']
    bound=max(max(float(abs(references[i]['phi']).max()),float(abs(predictions[i]['phi']).max())) for i in ids)
    error_max=max(float(abs(predictions[i]['phi']-references[i]['phi']).max()) for i in ids)
    for index in dict.fromkeys((ids[0],ids[len(ids)//2],ids[-1])):
        p=predictions[index];r=references[index]
        tri=mtri.Triangulation(p['mesh_points'][:,0],p['mesh_points'][:,1],p['mesh_cells'])
        fig,axes=plt.subplots(3,len(p['time']),figsize=(2.8*len(p['time']),7),layout='constrained')
        if plot_style=='smooth':
            display=SmoothMeshDisplay(p['mesh_points'],p['mesh_cells'],p['volumes'],p['holes'],
                                      extent=(0.,1.,.07,.93),resolution=(400,320))
            reference,prediction=display(r['phi']),display(p['phi'])
            arrays=(reference,prediction,np.ma.abs(prediction-reference))
        else:arrays=(r['phi'],p['phi'],abs(p['phi']-r['phi']))
        for row,values in enumerate(arrays):
            for column,t in enumerate(p['time']):
                ax=axes[row,column]
                options=dict(cmap='RdBu_r' if row<2 else 'magma',vmin=-bound if row<2 else 0,
                             vmax=bound if row<2 else error_max)
                if plot_style=='smooth':
                    im=ax.imshow(values[column],origin='lower',extent=display.extent,interpolation='nearest',**options)
                else:im=ax.tripcolor(tri,facecolors=values[column],shading='flat',rasterized=True,**options)
                for cx,cy,radius in p['holes']:
                    ax.add_patch(Circle((cx,cy),radius,fill=False,edgecolor='black',linewidth=.6))
                ax.set(aspect='equal',xlim=(0,1),ylim=(.07,.93));ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(f't = {t:.4g}')
                if column==0:ax.set_ylabel(('Native reference','Operator prediction','Absolute error')[row])
            fig.colorbar(im,ax=list(axes[row]),shrink=.8,pad=.01)
        fig.suptitle(f'Held-out geometry {index}: fixed ambient IC, variable impermeable holes\n{plot_style} display; metrics use original centroid fields')
        fig.savefig(output/f'fields_{index:03d}.png',dpi=180);plt.close(fig)
    records=training_records(run,'metrics.jsonl')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,field in zip(axes,('phi','mu')):
        for split in ('train','test'):
            ax.semilogy([r['step'] for r in records],[100*r[split+'_'+field+'_pooled'] for r in records],marker='o',label=split)
        ax.set(xlabel='Training updates',ylabel=f'{field} relative L2 (%)',title='Independent native reference')
        ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'convergence.png',dpi=180);plt.close(fig)
    rows=(len(ids)+3)//4
    fig,axes=plt.subplots(rows,min(4,len(ids)),figsize=(11,2.8*rows),layout='constrained',squeeze=False)
    initial_bound=max(float(abs(references[i]['phi'][0]).max()) for i in ids)
    for ax,index in zip(axes.ravel(),ids):
        r=references[index]
        tri=mtri.Triangulation(r['mesh_points'][:,0],r['mesh_points'][:,1],r['mesh_cells'])
        if plot_style=='smooth':
            display=SmoothMeshDisplay(r['mesh_points'],r['mesh_cells'],r['volumes'],r['holes'],
                                      extent=(0.,1.,.07,.93),resolution=(400,320))
            ax.imshow(display(r['phi'][:1])[0],origin='lower',extent=display.extent,
                      cmap='RdBu_r',vmin=-initial_bound,vmax=initial_bound,interpolation='nearest')
        else:ax.tripcolor(tri,facecolors=r['phi'][0],cmap='RdBu_r',vmin=-initial_bound,vmax=initial_bound)
        ax.set(aspect='equal',title=f'Test geometry {index}',xlim=(0,1),ylim=(.07,.93));ax.set_xticks([]);ax.set_yticks([])
    for ax in axes.ravel()[len(ids):]:ax.set_visible(False)
    fig.suptitle(f'Same initial function restricted to {len(ids)} unseen geometries')
    fig.savefig(output/'geometries.png',dpi=180);plt.close(fig)
    print(json.dumps({key:report[key] for key in ('train','test','local_FEM')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-style',choices=('flat','smooth'),default='smooth')
    args=parser.parse_args();evaluate(args.run,args.output,plot_style=args.plot_style)
