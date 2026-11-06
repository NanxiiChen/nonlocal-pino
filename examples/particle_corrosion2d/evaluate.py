"""Independent local-reference metrics and geometry-aware field galleries."""
from pathlib import Path
import json

import numpy as np


def load_references(directory,c):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    for name in ('geometry','particles','physical'):
        if json.loads(c.to_json())[name]!=manifest['config'][name]:
            raise ValueError(f'Reference {name} differs from training physics/prior')
    if manifest['config']['sample']['t_max']!=c.sample.t_max:
        raise ValueError('Reference time extent differs from training')
    references=[]
    for entry in manifest['cases']:
        with np.load(directory/entry['file'],allow_pickle=False) as source:
            data={key:source[key] for key in source.files}
        if (data['fields'].shape!=(len(data['times']),len(data['points']),2)
                or data['volumes'].shape!=(len(data['points']),)
                or np.any(data['volumes']<=0) or not np.isfinite(data['fields']).all()
                or not np.isfinite(data['volumes']).all()
                or data['points'].shape!=(len(data['volumes']),2) or not np.isfinite(data['points']).all()
                or data['times'].ndim!=1 or len(data['times'])<2 or not np.isfinite(data['times']).all()
                or np.any(np.diff(data['times'])<=0) or data['times'][0]!=0 or data['times'][-1]!=c.sample.t_max):
            raise ValueError('Invalid reference fields/physical quadrature')
        references.append(data)
    return references,manifest


def metrics(prediction,target,volumes,times):
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=2:
        raise ValueError('Predicted/reference physical fields must share(T,N,2)')
    tw=np.empty(len(times));tw[0]=(times[1]-times[0])/2;tw[-1]=(times[-1]-times[-2])/2
    tw[1:-1]=(times[2:]-times[:-2])/2
    weights=tw[:,None]*volumes[None,:]
    def relative(a,b,w):
        numerator=np.sum(w*(a-b)**2);denominator=np.sum(w*b*b)
        if denominator<=0: raise ValueError('Relative metric has a zero reference norm')
        return float(np.sqrt(numerator/denominator))
    result={}
    for i,name in enumerate(('phi','c')):
        result[name+'_pooled']=relative(prediction[...,i],target[...,i],weights)
        result[name+'_terminal']=relative(prediction[-1,:,i],target[-1,:,i],volumes)
    result['corroded_pooled']=relative(1-prediction[...,0],1-target[...,0],weights)
    result['corroded_terminal']=relative(1-prediction[-1,:,0],1-target[-1,:,0],volumes)
    result['phase_change']=relative(prediction[...,0]-target[0,:,0],target[...,0]-target[0,:,0],weights)
    a,b=prediction[-1,:,0]<.5,target[-1,:,0]<.5
    union=np.sum(volumes*(a|b))
    result['terminal_iou']=float(np.sum(volumes*(a&b))/union) if union>0 else 1.
    result['terminal_interface_area_error']=float(abs(np.sum(volumes*a)-np.sum(volumes*b))/np.sum(volumes*b))
    return result


def make_evaluator(c,references,chunk_size=512):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from .network import predict_fields
    names=('domain_tokens','domain_weights','boundary_tokens','boundary_weights')
    @eqx.filter_jit
    def infer(net,data,points,times):
        return predict_fields(net,data,points,times,c)
    inputs=[{name:jnp.asarray(ref[name],dtype=c.train.compute_dtype) for name in names} for ref in references]
    def evaluate(net):
        predictions=[];records=[]
        for ref,data in zip(references,inputs,strict=True):
            blocks=[]
            for offset in range(0,len(ref['points']),chunk_size):
                points=ref['points'][offset:offset+chunk_size]
                padded=np.pad(points,((0,chunk_size-len(points)),(0,0)),mode='edge')
                predicted=infer(net,data,jnp.asarray(padded,dtype=c.train.compute_dtype),
                    jnp.asarray(ref['times'],dtype=c.train.compute_dtype))
                blocks.append(np.asarray(jax.block_until_ready(predicted))[:,:len(points)])
            prediction=np.concatenate(blocks,axis=1)
            predictions.append(prediction)
            records.append(metrics(prediction,ref['fields'],ref['volumes'],ref['times']))
        summary={key:float(np.mean([record[key] for record in records])) for key in records[0]}
        summary['conditions']=records
        return predictions,summary
    return evaluate


def render_gallery(output,references,predictions,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    fig,axes=plt.subplots(len(references),5,figsize=(15,2.65*len(references)),squeeze=False)
    for row,(reference,prediction) in enumerate(zip(references,predictions,strict=True)):
        xy=reference['points'];tri=Triangulation(xy[:,0],xy[:,1],reference['triangles'])
        times=reference['times'];ids=np.rint(np.linspace(0,len(times)-1,5)).astype(int)
        for column,index in enumerate(ids):
            ax=axes[row,column]
            image=ax.tripcolor(tri,prediction[index,:,0],shading='gouraud',cmap='viridis',vmin=0,vmax=1,rasterized=True)
            if np.min(reference['fields'][index,:,0])<.5<np.max(reference['fields'][index,:,0]):
                ax.tricontour(tri,reference['fields'][index,:,0],levels=[.5],colors=['#ffdb73'],linestyles='--',linewidths=1.15)
            ax.set(xlim=(0,1),ylim=(.6,0),aspect='equal',title=f't*={times[index]:.2f}')
            ax.set_facecolor('#667680');ax.set_xticks([]);ax.set_yticks([])
            if column==0:ax.set_ylabel(f'Geometry {row+1}')
    fig.suptitle(title+' | dashed: local reference interface',fontsize=13)
    fig.subplots_adjust(left=.04,right=.94,top=.92,bottom=.05,wspace=.06,hspace=.16)
    fig.colorbar(image,cax=fig.add_axes([.955,.15,.012,.65]),label='metal phase phi')
    fig.savefig(output,dpi=150,facecolor='white',pil_kwargs={'optimize':True});plt.close(fig)


def render_convergence(path,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for key,label in [('phi_pooled','phi'),('c_pooled','concentration'),('corroded_pooled','corroded phase')]:
        axes[0].semilogy([r['step'] for r in history],[100*r[key] for r in history],label=label)
    for key,label in [('phi_terminal','phi'),('c_terminal','concentration'),('corroded_terminal','corroded phase')]:
        axes[1].semilogy([r['step'] for r in history],[100*r[key] for r in history],label=label)
    for ax,title in zip(axes,['Space-time error','Terminal error'],strict=True):
        ax.set(xlabel='Training updates',ylabel='Relative L2 (%)',title=title);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
