"""Independent checkpoint, raw-physics and interface audit of residual controls."""
from pathlib import Path
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from ml_collections import ConfigDict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import equinox as eqx
from examples.twophase2d.evaluate import audit_run
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.losses import make_residual_fn
from examples.twophase2d.problem import load_reference
OUT=None


def contour_geometry(phi, length):
    n=phi.shape[0];h=length/n
    if not phi.min()<0<phi.max():
        return dict(exists=False,perimeter=0.,components=None,width_proxy=None,points=None)
    axis=np.arange(n+1)*h
    fig,ax=plt.subplots()
    contour=ax.contour(axis,axis,np.pad(phi,((0,1),(0,1)),mode='wrap'),levels=[0])
    paths=contour.allsegs[0];plt.close(fig)
    samples=[];perimeter=0.;components=0
    for path in paths:
        if len(path)<2:continue
        cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        unique,ids=np.unique(cumulative,return_index=True)
        if unique[-1]<=0:continue
        path=path[ids];perimeter+=unique[-1];components+=1
        locations=np.arange(0,unique[-1],h/8)
        samples.append(np.stack([np.interp(locations,unique,path[:,k]) for k in range(2)],axis=-1)%length)
    if not samples:raise ValueError('A sign-changing phase field produced no contour')
    return dict(exists=True,perimeter=float(perimeter),components=components,
        width_proxy=float(np.mean(np.abs(phi)<.9)*length**2/perimeter),points=np.concatenate(samples))


def compare_geometry(phi, target, length):
    pred=contour_geometry(phi,length);ref=contour_geometry(target,length)
    if not ref['exists']:raise ValueError('Reference has no interface')
    result=dict(interface_exists=pred['exists'],perimeter=pred['perimeter'],
                reference_perimeter=ref['perimeter'],contour_pieces=pred['components'],
                width_proxy=pred['width_proxy'],reference_width_proxy=ref['width_proxy'])
    if pred['exists']:
        a,b=pred['points'],ref['points']
        da=cKDTree(b,boxsize=length).query(a)[0]
        db=cKDTree(a,boxsize=length).query(b)[0]
        result.update(symmetric_mean_contour_distance=float(.5*(da.mean()+db.mean())),
            symmetric_95percent_contour_distance=float(max(np.quantile(da,.95),np.quantile(db,.95))),
            width_proxy_relative_error=float(abs(pred['width_proxy']/ref['width_proxy']-1)))
    else:
        result.update(symmetric_mean_contour_distance=None,symmetric_95percent_contour_distance=None,
                      width_proxy_relative_error=None)
    return result


def validate_geometry():
    n=256;axis=np.arange(n)/n;X,Y=np.meshgrid(axis,axis,indexing='xy')
    phi=np.tanh((np.hypot(X-.5,Y-.5)-.2)/.02)
    shifted=np.tanh((np.hypot(X-.52,Y-.5)-.2)/.02)
    same=compare_geometry(phi,phi,1.);moved=compare_geometry(shifted,phi,1.)
    if same['symmetric_mean_contour_distance']!=0:raise ValueError('Geometry self-distance failed')
    if abs(moved['symmetric_mean_contour_distance']-.04/np.pi)>.001:raise ValueError('Shifted-circle distance failed')
    if abs(same['width_proxy']-2*.02*np.arctanh(.9))>.002:raise ValueError('Tanh-band width proxy failed')
    return dict(status='passed',shifted_circle_mean_distance=moved['symmetric_mean_contour_distance'],
                tanh_band_width_proxy=same['width_proxy'],definition='area(|phi|<0.9)/zero-contour perimeter; raster resolution proxy, not exact normal thickness')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    OUT=parser.parse_args().report_dir.resolve()
    if jax.default_backend()!='cpu':raise ValueError('Use JAX_PLATFORMS=cpu')
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Incomplete cohort')
    results=[];histories={};predictions=[]
    geometry_check=validate_geometry()
    for item in manifest['runs']:
        run=Path(item['run']);mode=item['mode'];folder=OUT/mode;folder.mkdir(exist_ok=True)
        result,history=audit_run(run,folder)
        c=ConfigDict(json.loads((run/'config.json').read_text()));reference=load_reference(c)
        with np.load(run/'prediction.npz') as saved:phi=saved['phi'];q=saved['q']
        predictions.append(phi);histories[mode]=history
        result['mode']=mode
        result['geometry_to_common_local']=[compare_geometry(a,b,c.geometry.length) for a,b in zip(phi,reference['phi_local'],strict=True)]
        # Original physical residuals are measured on identical21 times/full grids.
        with jax.enable_x64(False):template=PhaseNetwork(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
        model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
        points=jnp.asarray(reference['points'],dtype=jnp.float32);times=jnp.asarray(reference['time'],dtype=jnp.float32)
        D=PeriodicLaplacian(c.sample.Nx,c.geometry.length,c.nl.delta,dtype=np.float32)
        residual_fn=make_residual_fn(points,D,c)
        get_raw=eqx.filter_jit(lambda net:jax.vmap(residual_fn,in_axes=(None,0))(net,times)[0])
        raw=np.asarray(get_raw(model),dtype=np.float64).reshape(21,64,64,2)
        if 'phase' in c:
            get_jacobian=eqx.filter_jit(lambda net:jax.vmap(lambda t:jax.vmap(net.phase_jacobian,in_axes=(0,None))(points,t))(times))
            jacobian=np.asarray(get_jacobian(model),dtype=np.float64).reshape(phi.shape)
            w=(1+c.phase.leak)/jacobian
        else:
            w=np.sqrt(1+c.residual.eta**2)/np.sqrt(c.residual.eta**2+np.minimum((1-phi**2)**2,1))
        mask=np.abs(reference['phi_nonlocal'])<.9
        factors=np.ones_like(raw)
        if mode!='original':factors[...,0]=w
        if mode in ('both','map_both'):factors[...,1]=w
        weighted=raw*factors
        bulk=~mask
        result['raw_physics_audit']=dict(pooled_rms=np.sqrt(np.mean(raw**2,axis=(0,1,2))).tolist(),
            terminal_rms=np.sqrt(np.mean(raw[-1]**2,axis=(0,1))).tolist(),
            interface_rms=np.sqrt(np.mean(raw[mask]**2,axis=0)).tolist(),
            bulk_rms=np.sqrt(np.mean(raw[bulk]**2,axis=0)).tolist(),
            weight_max=float(w.max()),weight_median=float(np.median(w)),
            raw_bulk_squared_share=(np.sum(raw[bulk]**2,axis=0)/np.sum(raw**2,axis=(0,1,2))).tolist(),
            weighted_bulk_squared_share=(np.sum(weighted[bulk]**2,axis=0)/np.sum(weighted**2,axis=(0,1,2))).tolist())
        results.append(result)
        print(json.dumps(dict(mode=mode,common_local_midpoint=result['per_time_phi_relative_L2_to_local'][10],
            common_local_terminal=result['terminal_phi_relative_L2_to_local'],geometry=result['geometry_to_common_local'][-1],
            raw_physics=result['raw_physics_audit']),indent=2),flush=True)
    target=reference['phi_local'];ids=[0,5,10,15,20];axis=np.arange(64)/64;X,Y=np.meshgrid(axis,axis,indexing='xy')
    all_fields=[target]+predictions;labels=['Common local reference']+[r['mode'] for r in results]
    vmin=min(-1,min(float(p.min()) for p in all_fields));vmax=max(1,max(float(p.max()) for p in all_fields))
    fig,axes=plt.subplots(len(all_fields),5,figsize=(14,2.6*len(all_fields)),layout='constrained')
    for row,(field,label) in enumerate(zip(all_fields,labels,strict=True)):
        for col,i in enumerate(ids):
            ax=axes[row,col]
            image=ax.pcolormesh(X,Y,field[i],shading='nearest',cmap='viridis',vmin=vmin,vmax=vmax)
            if field[i].min()<0<field[i].max():ax.contour(X,Y,field[i],levels=[0],colors='white',linewidths=.7)
            if row:
                ax.contour(X,Y,target[i],levels=[0],colors='orangered',linewidths=.7,linestyles='dashed')
                err=np.linalg.norm(field[i]-target[i])/np.linalg.norm(target[i])
                ax.text(.02,.02,f'L2={100*err:.2f}%',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
            if row==0:ax.set_title(f't={reference["time"][i]:.2f}')
            if col==0:ax.set_ylabel(label+'\ny')
            if row==len(all_fields)-1:ax.set_xlabel('x')
            ax.set(aspect='equal')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='phi (full range, no clipping)',shrink=.8)
    fig.suptitle('Same physical model/initial network/5000 updates: residual-preconditioning controls')
    fig.savefig(OUT/'fields.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for r in results:
        mode=r['mode'];h=histories[mode]
        axes[0].semilogy([v['step'] for v in h],[v['terminal_rel_l2_phi_local'] for v in h],label=mode)
        axes[1].plot(reference['time'],r['per_time_phi_relative_L2_to_local'],label=mode)
        axes[2].plot(reference['time'],[v['symmetric_mean_contour_distance'] for v in r['geometry_to_common_local']],label=mode)
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    axes[0].set(xlabel='Updates',ylabel='Terminal phi L2 to common local reference')
    axes[1].set(xlabel='Physical time',ylabel='Phi L2 to common local reference')
    axes[2].set(xlabel='Physical time',ylabel='Mean symmetric contour distance')
    fig.savefig(OUT/'convergence.png',dpi=170);plt.close(fig)
    (OUT/'analysis.json').write_text(json.dumps(dict(status='passed',geometry_self_checks=geometry_check,runs=results),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
