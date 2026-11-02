"""Saved-field diagnostic: evolving droplet versus original hard-IC footprint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.problem import initial_field,walls


def diagnose(paths,labels,output):
    if len(paths)!=len(labels)or len(set(labels))!=len(labels):raise ValueError('Unique aligned field labels required')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    results=[];common=None
    for path,label in zip(map(Path,paths),labels,strict=True):
        c=ConfigDict(json.loads((path.parent/'config.json').read_text()))
        with np.load(path,allow_pickle=False)as f:d={k:f[k]for k in f.files}
        x=d['points'];v=d['fields'][...,0];ref=d['reference'][...,0]
        if v.shape!=ref.shape or not np.isfinite(v).all()or not np.isfinite(ref).all():raise ValueError('Finite aligned phase fields required')
        if common is None:common=d
        else:
            for key in ('points','time','reference','amplitudes'):np.testing.assert_array_equal(d[key],common[key])
        nx=len(np.unique(x[:,0]));ny=len(x)//nx
        bottom,top=walls(x[:,0],d['amplitudes'],c,xp=np)
        measure=(top-bottom)*c.geometry.length/(nx*ny)
        ic=initial_field(x,c,xp=np);initial=ic<.9;rows=[]
        for i,t in enumerate(d['time']):
            if t==0:continue
            droplet=ref[i]<.9;old=initial&~droplet;rest=~(droplet|old)
            masks=(droplet,old,rest);names=('current_droplet_band','old_IC_only','remaining_domain')
            np.testing.assert_array_equal(sum(m.astype(int)for m in masks),np.ones(len(x),dtype=int))
            error=measure*(v[i]-ref[i])**2;total=error.sum()
            if total<=0:raise ValueError('Nonzero field error required for an error-fraction diagnostic')
            regions=[]
            for name,mask in zip(names,masks,strict=True):
                area=np.sum(measure*mask);square=float(error[mask].sum())
                regions.append(dict(region=name,points=int(mask.sum()),area=float(area),
                    area_fraction=float(area/measure.sum()),squared_error=square,
                    squared_error_fraction=float(square/total),RMS=float(np.sqrt(square/area))if area>0 else None))
            np.testing.assert_allclose(sum(a['squared_error']for a in regions),total,rtol=2e-14,atol=1e-16)
            thin=np.abs(ref[i])<.9;core=ref[i]<=-.9
            np.testing.assert_array_equal(thin|core,droplet)
            rows.append(dict(time=float(t),total_squared_error=float(total),regions=regions,
                thin_interface_squared_error_fraction=float(error[thin].sum()/total),
                droplet_core_squared_error_fraction=float(error[core].sum()/total)))
            components={}
            for component,column in (('u_x',2),('u_y',3)):
                target=d['reference'][i,:,column];delta=d['fields'][i,:,column]-target
                denominator=np.sum(measure*target*target)
                if denominator<=0:raise ValueError('Positive-time reference velocity component has zero norm')
                components[component]=dict(relative_L2=float(np.sqrt(np.sum(measure*delta*delta)/denominator)),
                    absolute_RMS=float(np.sqrt(np.sum(measure*delta*delta)/measure.sum())),
                    reference_RMS=float(np.sqrt(denominator/measure.sum())))
            rows[-1]['velocity_components']=components
        results.append(dict(label=label,fields=str(path),rows=rows))
    output.mkdir(parents=True)
    report=dict(status='saved_phase_error_regions_diagnosed',results=results,
        training_updates=0,reference_solves=0,
        scope='Area-weighted squared phase error partition only; reference/knownIC masks never enter training. Different checkpoint budgets are explicitly labeled; no causal or architecture lower-bound claim.')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');left=np.zeros(len(results))
    for j,(label,color)in enumerate(zip(('Current interface','Current droplet core','Old IC footprint only','Remaining domain'),('#0072B2','#56B4E9','#D55E00','#999999'),strict=True)):
        def fraction(row):
            if j==0:return row['thin_interface_squared_error_fraction']
            if j==1:return row['droplet_core_squared_error_fraction']
            return row['regions'][j-1]['squared_error_fraction']
        size=np.array([fraction(r['rows'][-1])for r in results])*100
        ax.barh(np.arange(len(results)),size,left=left,label=label,color=color);left+=size
    ax.set(yticks=np.arange(len(results)),yticklabels=labels,xlim=(0,100),xlabel='Fraction of terminal phase squared error (%)',title='Saved phase-error location: diagnostic masks only')
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,-.5),ncol=2,fontsize=8)
    fig.savefig(output/'terminal-error-regions.png',dpi=200,bbox_inches='tight',pad_inches=.08)
    fig.savefig(output/'terminal-error-regions.pdf',bbox_inches='tight',pad_inches=.08);plt.close(fig)
    print(json.dumps([dict(label=r['label'],terminal=r['rows'][-1])for r in results],indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fields',nargs='+',required=True)
    p.add_argument('--labels',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();diagnose(a.fields,a.labels,a.output)
