"""Compare geometry-induced field variations on a common physical probe set."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from ml_collections import ConfigDict
from examples.confined_ch2d.geometry import visible_pairs
from examples.confined_ch2d.operator_problem import fixed_initial


def reference_probes(root):
    c=ConfigDict(json.loads((root/'config.json').read_text()))
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='passed':raise ValueError('Validated reference bank required')
    ids=manifest['test_indices'];refs=[]
    axisx=np.linspace(.015,.985,120);axisy=np.linspace(.115,.885,96)
    grid=np.stack(np.meshgrid(axisx,axisy,indexing='xy'),axis=-1)
    points=grid.reshape(-1,2)
    keep=np.sum(((points-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=1)<.96**2
    mappings=[]
    for index in ids:
        with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as f:
            d={key:f[key] for key in f.files}
        refs.append(d)
        tri=Delaunay(d['points']);simplices=tri.simplices
        edges=np.stack((simplices[:,[0,1]],simplices[:,[1,2]],simplices[:,[2,0]]),axis=1)
        good=np.all(visible_pairs(d['points'],edges.reshape(-1,2),d['holes']).reshape(-1,3),axis=1)
        # Also exclude a triangle enclosing an entire hole without edge crossing.
        centers=tri.find_simplex(d['holes'][:,:2]);good[centers[centers>=0]]=False
        owner=tri.find_simplex(points);safe=np.maximum(owner,0)
        valid=(owner>=0)&good[safe]
        for cx,cy,r in d['holes']:
            valid &= np.sum((points-np.array([cx,cy]))**2,axis=1)>r*r
        keep &= valid
        delta=points-tri.transform[safe,2]
        first=np.einsum('nij,nj->ni',tri.transform[safe,:2],delta)
        bary=np.column_stack((first,1-first.sum(axis=1)))
        mappings.append((simplices[safe],bary))
    if keep.sum()<100:raise ValueError('Too few common physical probes')
    fields=[];raw_initial=[]
    common=points[keep];initial=fixed_initial(common,c,xp=np)
    for d,(vertices,bary) in zip(refs,mappings,strict=True):
        vertices,bary=vertices[keep],bary[keep]
        interpolated=np.einsum('tnk,nk->tn',d['phi'][:,vertices],bary)
        raw_initial.append(interpolated[0])
        # Same exact ambient IC removes interpolation-induced initial variance.
        # Only evolution increments are interpolated, through visible fluid triangles.
        fields.append(initial[None,:]+interpolated-interpolated[0])
    return c,manifest,refs,grid,keep,common,np.stack(fields),np.stack(raw_initial)


def variation(a):
    mean=a.mean(axis=0);centered=a-mean
    spread=np.sqrt(np.mean(np.sum(centered**2,axis=-1),axis=0))
    return dict(relative_spread_by_time=(spread/np.linalg.norm(mean,axis=-1)).tolist(),
        terminal_pointwise_std_max=float(a[:,-1].std(axis=0).max()),
        terminal_pointwise_std_rms=float(np.sqrt(np.mean(a[:,-1].var(axis=0)))))


def diagnose(root,output,run):
    output.mkdir(parents=True,exist_ok=False)
    c,manifest,refs,grid,keep,points,truth,raw_initial=reference_probes(root)
    result=dict(dataset=str(root),indices=manifest['test_indices'],common_points=len(points),
        scope='Common fluid intersection; piecewise linear native evolution increments on centroid triangles with hole-crossing stencils removed; not an exact common-grid FEM solution',
        reference=variation(truth),
        removed_initial_interpolation_std_rms=float(np.sqrt(np.mean(raw_initial.var(axis=0)))),
        maximum_reference_absolute_phi=float(max(abs(d['phi']).max() for d in refs)))
    prediction=None
    if run is not None:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from examples.confined_ch2d.operator_training import build_network,load_bank,query_phase
        jax.config.update('jax_enable_x64',True)
        saved=ConfigDict(json.loads((run/'config.json').read_text()))
        if saved.operator.dataset_path!=str(root):raise ValueError('Checkpoint and diagnostic bank must match')
        _,_,bank=load_bank(saved)
        model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(saved,jax.random.key(0)))
        predict=eqx.filter_jit(lambda n,d,x,t:query_phase(n,d,x,t,saved))
        prediction=[]
        for index,d in zip(manifest['test_indices'],refs,strict=True):
            item=jax.tree.map(lambda a:a[index],bank)
            prediction.append(np.asarray(predict(model,item,jnp.asarray(points),jnp.asarray(d['time']))))
        prediction=np.stack(prediction)
        centered=prediction-prediction.mean(axis=0)
        actual=truth-truth.mean(axis=0)
        actual_norm=np.linalg.norm(actual);predicted_norm=np.linalg.norm(centered)
        if actual_norm==0:
            raise ValueError('Relative geometry-response error requires nonzero reference variation')
        result.update(run=str(run),prediction=variation(prediction),
            geometry_response_relative_error=float(np.linalg.norm(centered-actual)/np.linalg.norm(actual)),
            geometry_response_amplitude_ratio=float(predicted_norm/actual_norm),
            # A constant prediction has no defined response direction/correlation.
            geometry_response_correlation=float(np.sum(centered*actual)/(predicted_norm*actual_norm)) if predicted_norm>0 else None,
            whole_field_relative_error=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth)),
            mean_field_relative_error=float(np.linalg.norm(prediction.mean(axis=0)-truth.mean(axis=0))/np.linalg.norm(truth.mean(axis=0))))
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[(truth[:,-1].mean(axis=0),'Reference: mean phase','RdBu_r'),
            (truth[:,-1].std(axis=0),'Reference: geometry spread','magma')]
    if prediction is not None:
        panels.append((prediction[:,-1].std(axis=0),'Prediction: geometry spread','magma'))
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),4),layout='constrained')
    spread_max=max(float(panel[0].max()) for panel in panels[1:])
    for ax,(values,title,cmap) in zip(axes,panels,strict=True):
        image=np.full(len(keep),np.nan);image[keep]=values
        bound=max(abs(values)) if cmap=='RdBu_r' else spread_max
        im=ax.imshow(image.reshape(grid.shape[:2]),origin='lower',extent=(.015,.985,.115,.885),
            cmap=cmap,vmin=-bound if cmap=='RdBu_r' else 0,vmax=bound)
        ax.set(title=title,aspect='equal');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Geometry response at identical coordinates; white = excluded common support')
    fig.savefig(output/'geometry_spread.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args();diagnose(args.dataset,args.output,args.run)
