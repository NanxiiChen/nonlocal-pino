"""Final-budget phase fields at low, median and worst validation errors."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from examples.confined_ch2d.operator_plotting import SmoothMeshDisplay
from docs.tasks.confined_ch_random_gallery import relative_l2


def gallery(run,output):
    if not run.is_absolute() or not output.is_absolute():raise ValueError('Absolute run/output paths required')
    c=json.loads((run/'config.json').read_text());s=json.loads((run/'summary.json').read_text())
    if s['start_step']!=0 or s['new_updates']!=50000 or s['final_metrics']['step']!=50000:
        raise ValueError('Require the completed fresh50000update result')
    bank=Path(c['operator']['dataset_path']);m=json.loads((bank/'manifest.json').read_text())
    ids=m['test_indices'];scores={a['index']:a for a in s['final_metrics']['conditions']}
    if m['status']!='passed' or len(ids)!=8 or set(ids)&set(m['train_indices']):
        raise ValueError('Require eight disjoint reserved geometries')
    records={};verified=[];bound=0.;error_bound=0.
    initial_tolerance=float(4*np.finfo(np.dtype(c['train']['compute_dtype'])).eps)
    for i in ids:
        with np.load(run/f'prediction_{i:03d}.npz',allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        reference=bank/m['conditions'][i]['file']
        with np.load(reference,allow_pickle=False) as f:
            for key in ['points','volumes','holes','time','mesh_points','mesh_cells']:
                np.testing.assert_array_equal(d[key],f[key])
            np.testing.assert_array_equal(d['phi_reference'],f['phi'])
        np.testing.assert_array_equal(d['time'],[0.,.75,1.5,2.25,3.])
        actual,target,v=d['phi'],d['phi_reference'],d['volumes']
        if actual.shape!=(5,len(v)) or target.shape!=actual.shape or np.any(v<=0):
            raise ValueError(f'Invalid shape/quadrature for{i}')
        if not all(np.isfinite(a).all() for a in [actual,target,v]):raise ValueError('Nonfinite physical fields')
        initial_gap=float(abs(actual[0]-target[0]).max())
        if initial_gap>initial_tolerance:raise ValueError('HardIC mismatch exceeds compute rounding tolerance')
        pooled=relative_l2(actual,target,v);terminal=relative_l2(actual[-1],target[-1],v)
        np.testing.assert_allclose([pooled,terminal],[scores[i]['phi_pooled'],scores[i]['phi_terminal']],rtol=1e-11,atol=1e-14)
        verified.append(dict(index=i,phi_pooled=pooled,phi_terminal=terminal,nodes=len(v),
            initial_max_gap=initial_gap,reference=str(reference)))
        bound=max(bound,float(abs(actual).max()),float(abs(target).max()))
        error_bound=max(error_bound,float(abs(actual-target).max()))
        records[i]=d
    order=sorted(ids,key=lambda i:(scores[i]['phi_pooled'],i))
    selected=[order[0],order[len(order)//2],order[-1]]
    labels=['Low-error case','Upper-median error case','Maximum-error case']
    if len(set(selected))!=3 or min(bound,error_bound)<=0:raise ValueError('Invalid display selection/scales')
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'pdf.fonttype':42,'figure.facecolor':'white'})
    figures=[]
    for i,label in zip(selected,labels,strict=True):
        d=records[i];display=SmoothMeshDisplay(d['mesh_points'],d['mesh_cells'],d['volumes'],d['holes'],
            extent=(0.,1.,.07,.93),resolution=(600,480))
        fields=[display(d['phi_reference']),display(d['phi']),display(abs(d['phi']-d['phi_reference']))]
        for a in fields:
            if not np.isfinite(a.compressed()).all():raise ValueError('Nonfinite display interpolation')
        if fields[2].min()<0 or fields[2].max()>error_bound*(1+1e-12):raise ValueError('Display changes absolute-error bounds')
        fig,axes=plt.subplots(3,5,figsize=(14.6,7.7),layout='constrained')
        for row,values in enumerate(fields):
            for col,t in enumerate(d['time']):
                ax=axes[row,col]
                im=ax.imshow(values[col],origin='lower',extent=display.extent,interpolation='nearest',
                    cmap='RdBu_r' if row<2 else 'magma',vmin=-bound if row<2 else 0,
                    vmax=bound if row<2 else error_bound)
                if row<2:
                    ax.contour(display.xx,display.yy,values[col],levels=[0.],colors='black',linewidths=.35,alpha=.5)
                for cx,cy,radius in d['holes']:
                    ax.add_patch(Circle((cx,cy),radius,facecolor='white',edgecolor='black',lw=.6))
                ax.set(aspect='equal',xticks=[],yticks=[])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:ax.set_title(f't = {t:.2f}')
                if col==0:ax.set_ylabel(['Nonlocal reference','Operator prediction','Absolute error'][row],labelpad=12)
            fig.colorbar(im,ax=list(axes[row]),shrink=.85,pad=.012,label='Phase phi' if row<2 else 'Absolute phase error')
        score=scores[i]
        fig.suptitle(f"{label}: validation geometry {i} | phase L2 {score['phi_pooled']:.2%}, final {score['phi_terminal']:.2%}\n"
            'Final 50k model; common phase/error color scales across all 8 validation geometries',fontsize=12)
        for suffix in ['png','pdf']:fig.savefig(output/f'fields_{i:04d}.{suffix}',dpi=190)
        plt.close(fig)
        figures.append(dict(index=i,label=label,phi_pooled=score['phi_pooled'],phi_terminal=score['phi_terminal'],
            png=str(output/f'fields_{i:04d}.png'),pdf=str(output/f'fields_{i:04d}.pdf')))
    result=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),run=str(run),step=50000,
        model='model_final.eqx',selection='Low, uppermedian andmaximum pooled validationerror; explicitly stratified, not random or checkpoint selection',
        selected=selected,figures=figures,verified_all8=verified,initial_gap_tolerance=initial_tolerance,
        phase_bound=bound,absolute_error_bound=error_bound,times=[0.,.75,1.5,2.25,3.],
        display='Area-weight incident-cell projection, linear interpolation on original triangles, analytic hole masks; raw absolute error interpolated directly. Display only, unchanged physical-cell L2.',
        reference='Native regional graph/BDF, not localFEM',scope='Same8reused development geometries, fixed-budget final50k; no new test or all-terminal-errors-below10% claim')
    (output/'gallery.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();gallery(args.run,args.output)
