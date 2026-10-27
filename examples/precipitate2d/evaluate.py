"""Replay a direct checkpoint path and plot physical fields/convergence."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict


def evaluate(run,checkpoint):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from examples.precipitate2d.problem import load_reference,initial_field,validate_config
    from examples.precipitate2d.train import _build_network
    from examples.precipitate2d.network import predict_fields
    from examples.precipitate2d.mechanics import ElasticEquilibrium
    from examples.precipitate2d.solver import relative
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    validate_config(c)
    jax.config.update('jax_enable_x64',True)
    ref = load_reference(c)
    kn,_ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    # Serialized leaves replace every trainable leaf; static fields use config.
    net = eqx.tree_deserialise_leaves(checkpoint,_build_network(c,kn))
    x = jnp.asarray(ref['points'])
    mean = jnp.mean(initial_field(x,c,xp=jnp))
    mech = ElasticEquilibrium(c)
    @eqx.filter_jit
    def at(t):
        pq = predict_fields(net,x,t,mean)
        _,u,s = mech(pq[:,0])
        return pq,u,s
    n,nt = c.sample.Nx,len(ref['time'])
    pq,u,s = (np.asarray(a) for a in jax.vmap(at)(jnp.asarray(ref['time'])))
    fields = dict(phi=pq[...,0].reshape(nt,n,n),q=pq[...,1].reshape(nt,n,n),
                  u=u.reshape(nt,n,n,2),stress=s.reshape(nt,n,n,3))
    kind = 'nonlocal' if c.nl.method=='nonlocal' else 'local'
    report = dict(checkpoint=str(checkpoint),reference=str(c.reference.path),method=c.nl.method)
    for key,a in fields.items():
        b = ref[key+'_'+kind]
        report['pooled_rel_l2_'+key] = relative(a,b)
        report['terminal_rel_l2_'+key] = relative(a[-1],b[-1])
    report['pooled_rel_l2_phi_local'] = relative(fields['phi'],ref['phi_local'])
    report['mass_max_drift'] = float(np.max(np.abs(fields['phi'].mean(axis=(1,2))-float(mean))))
    if checkpoint.name=='model_best.eqx':
        with np.load(run/'prediction_best.npz') as saved:
            for key,a in fields.items():
                np.testing.assert_allclose(a,saved[key],atol=1e-8,rtol=1e-8,err_msg='Direct checkpoint replay mismatch: '+key)
        recorded = json.loads((run/'best_metrics.json').read_text())
        for key in ('phi','q','u','stress'):
            np.testing.assert_allclose(report['pooled_rel_l2_'+key],recorded['pooled_rel_l2_'+key],atol=1e-8,rtol=1e-8)
    np.savez_compressed(run/'replayed_prediction.npz',time=ref['time'],points=ref['points'],**fields)
    (run/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    selected = np.linspace(0,nt-1,5).round().astype(int)
    length = c.geometry.length
    h = length/n
    coords = np.arange(n)*h
    def show(ax,a,**kwargs):
        im = ax.imshow(np.pad(a,((0,1),(0,1)),mode='wrap'),origin='lower',
            extent=(-h/2,length+h/2,-h/2,length+h/2),**kwargs)
        ax.set_xlim(0,length);ax.set_ylim(0,length)
        return im
    target = ref['phi_'+kind]
    error = np.abs(fields['phi']-target)
    fig,axes = plt.subplots(3,5,figsize=(13,7),layout='constrained')
    for col,i in enumerate(selected):
        for row,a in enumerate((target[i],fields['phi'][i],error[i])):
            im = show(axes[row,col],a,
                cmap='magma' if row==2 else 'coolwarm',vmin=0 if row==2 else -1,
                vmax=max(error.max(),1e-15) if row==2 else 1)
            if row<2:axes[row,col].contour(coords,coords,target[i],levels=[0],colors='k',linewidths=.6)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[0,col].set_title(f't = {ref["time"][i]:.2f}')
    for row,label in enumerate(('Reference phi','PINN phi','Absolute error')):axes[row,0].set_ylabel(label)
    fig.colorbar(axes[0,0].images[0],ax=axes[:2,:].ravel().tolist(),shrink=.7,label='phi')
    fig.colorbar(im,ax=axes[2,:].tolist(),shrink=.7,label='|error|')
    fig.savefig(run/'phi_fields.png',dpi=180);plt.close(fig)

    fig,axes = plt.subplots(2,5,figsize=(13,5.5),layout='constrained')
    for row,(key,label,component) in enumerate((('u','Displacement magnitude',None),('stress','Stress xx',0))):
        pred = np.linalg.norm(fields[key],axis=-1) if component is None else fields[key][...,component]
        truth = np.linalg.norm(ref[key+'_'+kind],axis=-1) if component is None else ref[key+'_'+kind][...,component]
        lo,hi = float(min(pred.min(),truth.min())),float(max(pred.max(),truth.max()))
        for col,i in enumerate(selected):
            im = show(axes[row,col],pred[i],cmap='viridis',vmin=lo,vmax=hi)
            axes[row,col].contour(coords,coords,fields['phi'][i],levels=[0],colors='w',linewidths=.6)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f't = {ref["time"][i]:.2f}')
        axes[row,0].set_ylabel(label)
        fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.8)
    fig.savefig(run/'mechanical_fields.png',dpi=180);plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(9,4),layout='constrained')
    history = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    for key in ('phi','u','stress','q'):
        axes[0].semilogy([r['step'] for r in history],[max(r['pooled_rel_l2_'+key],1e-16) for r in history],label=key)
    axes[0].set_xlabel('Optimizer updates');axes[0].set_ylabel('Pooled relative L2');axes[0].legend();axes[0].grid(alpha=.3)
    for a,label,color in [(target[0],'Initial','gray'),(ref['phi_uncoupled'][-1],'No elasticity','C1'),
                           (target[-1],'Coupled reference','C0'),(fields['phi'][-1],'PINN','C2')]:
        axes[1].contour(coords,coords,a,levels=[0],colors=[color])
        axes[1].plot([],[],color=color,label=label)
    axes[1].set_aspect('equal');axes[1].set_title('Terminal zero contours');axes[1].legend(fontsize=8)
    if 'kind' in c.initial:
        axes[1].set_xlim(0,length);axes[1].set_ylim(0,length)
    else:
        axes[1].set_xlim(c.initial.center[0]-.26*length,c.initial.center[0]+.26*length)
        axes[1].set_ylim(c.initial.center[1]-.26*length,c.initial.center[1]+.26*length)
    fig.savefig(run/'convergence_and_shape.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,help='Explicit checkpoint; otherwise run/model_best.eqx')
    args = parser.parse_args()
    evaluate(args.run,args.run/'model_best.eqx' if args.checkpoint is None else args.checkpoint)
