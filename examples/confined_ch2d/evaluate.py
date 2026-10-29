"""Direct saved-config checkpoint replay and unstructured physical-field plots."""
from pathlib import Path
import sys,argparse,json
if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def field_plot(ref,fields,path,labels):
    """Cell quadrature fields plotted on the actual conforming mesh, including holes."""
    tri=mtri.Triangulation(ref['mesh_points'][:,0],ref['mesh_points'][:,1],ref['mesh_cells'])
    indices=np.linspace(0,len(ref['time'])-1,5).round().astype(int)
    fig,axes=plt.subplots(len(fields),5,figsize=(13,2.5*len(fields)),layout='constrained',squeeze=False)
    for row,(values,label) in enumerate(zip(fields,labels,strict=True)):
        error='error' in label.lower()
        lo,hi=(0,max(float(values.max()),1e-15)) if error else (-1,1)
        for col,i in enumerate(indices):
            im=axes[row,col].tripcolor(tri,facecolors=values[i],cmap='magma' if error else 'coolwarm',vmin=lo,vmax=hi,rasterized=True)
            axes[row,col].set(aspect='equal');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f't={ref["time"][i]:.2f}')
            if col==0:axes[row,col].set_ylabel(label)
        fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.75)
    fig.savefig(path,dpi=180);plt.close(fig)


def evaluate(run,checkpoint):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.confined_ch2d.train import _build_network
    from examples.confined_ch2d.problem import load_reference,initial_field,boundary_samples,relative
    from examples.confined_ch2d.network import predict_fields
    from examples.confined_ch2d.operators import RegionalLaplacian
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if 'windows' in c.train:
        from examples.confined_ch2d.window_evaluate import evaluate_windows
        return evaluate_windows(run,checkpoint)
    jax.config.update('jax_enable_x64',True)
    ref=load_reference(c);x=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes'])
    mean=jnp.dot(v,initial_field(x,c,xp=jnp))/v.sum()
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(checkpoint,_build_network(c,key))
    D=RegionalLaplacian(ref['points'],ref['volumes'],c) if net.chemical_mode=='eliminated' else None
    @eqx.filter_jit
    def eval_fields(net):return jax.vmap(lambda t:predict_fields(net,x,t,mean,v,D))(jnp.asarray(ref['time']))
    values=np.asarray(eval_fields(net));fields=dict(phi=values[...,0],mu=values[...,1])
    saved_name='prediction_best.npz' if checkpoint.name=='model_best.eqx' else 'prediction.npz'
    if checkpoint.name not in ('model_best.eqx','model_final.eqx'):raise ValueError('Replay accepts explicit best/final checkpoints')
    errors={}
    with np.load(run/saved_name) as saved:
        for k,a in fields.items():
            np.testing.assert_allclose(a,saved[k],atol=1e-8,rtol=1e-8);errors[k]=float(np.max(np.abs(a-saved[k])))
    report=dict(checkpoint=str(checkpoint),cpu_replay_max_absolute=errors,common_local={})
    for key,a in fields.items():report['common_local'][key]=dict(pooled=relative(a,ref[key+'_local'],ref['volumes']),terminal=relative(a[-1],ref[key+'_local'][-1],ref['volumes']))
    report['mass_max_drift']=float(np.max(np.abs(fields['phi']@ref['volumes']/ref['volumes'].sum()-float(mean))))
    boundary,normals,bw,wall=boundary_samples(c)
    @eqx.filter_jit
    def normal_derivatives(net):
        def at(t):
            from examples.confined_ch2d.network import TanhPhaseNetwork
            if isinstance(net,TanhPhaseNetwork):
                from examples.confined_ch2d.tanh_mapping import phase_state,stable_tanh
                _,_,offset=phase_state(net,x,t,mean,v)
                physical=lambda point:jnp.atleast_1d(net.tanh_amplitude*stable_tanh(net.latent(point,t)+offset))
                jac=jax.vmap(jax.jacfwd(physical))(jnp.asarray(boundary))
            else:jac=jax.vmap(jax.jacfwd(net,argnums=0),in_axes=(0,None))(jnp.asarray(boundary),t)
            return jnp.einsum('nfc,nc->nf',jac,jnp.asarray(normals))
        return jax.vmap(at)(jnp.asarray(ref['time']))
    deriv=np.asarray(normal_derivatives(net))
    report['wall_normal_fields']=['phi'] if net.chemical_mode=='eliminated' else ['phi','mu']
    if net.chemical_mode=='eliminated':report['chemical_normal_scope']='Native mu is defined at quadrature points; no classical boundary extension is supplied'
    report['wall_normal_rms']={str(i):np.sqrt(np.sum(deriv[:,wall==i,:]**2*bw[wall==i][None,:,None],axis=1)/bw[wall==i].sum()).tolist() for i in np.unique(wall)}
    # Native nonlocal normal derivatives are LOCAL-LIMIT diagnostics, not zero constraints.
    wall_distance=np.abs(.5*min(c.geometry.axes)*(1-np.sum(((ref['points']-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=1)))
    for cx,cy,r in c.geometry.holes:wall_distance=np.minimum(wall_distance,np.abs(np.linalg.norm(ref['points']-np.array([cx,cy]),axis=1)-r))
    band=wall_distance<c.nl.delta
    report['boundary_band_phi_local']=relative(fields['phi'][:,band],ref['phi_local'][:,band],ref['volumes'][band])
    (run/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(run/'replayed_prediction.npz',**fields)
    field_plot(ref,[ref['phi_local'],fields['phi'],np.abs(fields['phi']-ref['phi_local'])],run/'phi_fields.png',['Common local reference','PINN phi','Absolute error'])
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for k in ('phi','mu'):ax.semilogy([s['step'] for s in history],[s['pooled_rel_l2_'+k+'_local'] for s in history],label=k)
    ax.set(xlabel='Optimizer iterations',ylabel='Pooled relative L2, common local target');ax.legend();ax.grid(alpha=.3)
    fig.savefig(run/'convergence.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();evaluate(args.run,args.checkpoint)
