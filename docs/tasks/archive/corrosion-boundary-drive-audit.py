"""Read-only CPU boundary/initial-drive and frozen BC sensitivity audit."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np
from scipy import sparse
from examples.corrosion2d.residual_experiment import build_model, boundary_values, initial_fields, load_reference
from examples.corrosion2d.train import _bc_info

ROOT = Path('/root/autodl-tmp/tf-logs/corrosion2d')
RUNS = {
    'accurate original': ROOT/'20261025-residual-phase-control-seed42',
    'collapsed original': ROOT/'20261025-stagger0-every25-phase-mapkks-ac1e-07-ch1.0-seed42',
    'collapsed latent stagger': ROOT/'20261025-stagger1000-every25-latent-mapkks-ac1e-07-ch1.0-seed42',
}


def h(p):
    return 3*p**2-2*p**3


def rms(a):
    return float(np.sqrt(np.mean(a*a)))


def stats(a):
    return {'rms': rms(a), 'max_abs': float(np.max(np.abs(a))),
            'min': float(a.min()), 'max': float(a.max())}


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Set JAX_PLATFORMS=cpu')
    data = {'scope': 'Saved checkpoints, split BC/IC/drive metrics; frozen clamps are not training', 'runs': {}}
    profiles = {}
    for label, run in RUNS.items():
        cfg=ConfigDict(json.loads((run/'config.json').read_text()))
        ref=load_reference(cfg)
        dc,cle=cfg.physical.c_se-cfg.physical.c_le,cfg.physical.c_le
        points=np.stack([ref['X'].ravel(),ref['Y'].ravel()],axis=-1)
        zero,one,_=boundary_values(cfg,points);free=~(zero|one)
        row,col,weight=(ref[k] for k in ('graph_row','graph_col','graph_weight'))
        n=len(points)
        matrix=sparse.diags(np.bincount(row,weights=weight,minlength=n))-sparse.csr_matrix((weight,(row,col)),shape=(n,n))
        np.testing.assert_allclose(matrix@np.ones(n),0,atol=1e-8)
        targetphi,targetc=ref['phi'].reshape(21,n),ref['c'].reshape(21,n)
        targetq=targetc-dc*h(targetphi)-cle
        analytic=initial_fields(cfg,points).astype(np.float64)
        qic=analytic[:,1]-dc*h(analytic[:,0])-cle
        # FP32 analytic IC values are checked with their actual construction tolerance.
        np.testing.assert_allclose(qic,cle*(h(analytic[:,0])-1),atol=3e-7,rtol=0)
        np.testing.assert_allclose(targetq[:,zero],-cle,atol=1e-14)
        np.testing.assert_allclose(targetq[:,one],0,atol=1e-14)
        arc=(targetphi[0]>.05)&(targetphi[0]<.95)&free
        pit,solid=_bc_info(cfg.pits.entries,cfg.sample.N_bc_dir,cfg.sample.N_bc_solid)
        allpoints=jnp.asarray(np.concatenate([points,np.asarray(pit),np.asarray(solid)]),dtype=jnp.float32)
        times=jnp.asarray(ref['time'][[0,1,-1]],dtype=jnp.float32)
        @eqx.filter_jit
        def evaluate(net):
            raw=jax.vmap(lambda t:jax.vmap(net.net,in_axes=(0,None))(allpoints,t))(times)
            return raw,net.constraint(raw)
        key=jax.random.split(jax.random.PRNGKey(cfg.train.seed))[0]
        records={}
        for step in (1000,5000):
            path=run/f'ckpt_{step:06d}.eqx'
            net=eqx.tree_deserialise_leaves(path,build_model(cfg,key))
            raw,physical=(np.asarray(v) for v in evaluate(net))
            phi=physical[...,0].astype(np.float64);c=physical[...,1].astype(np.float64)
            cl=cle*.5*(1+np.tanh(raw[...,1].astype(np.float64)))
            q=cl-cle
            # Stable raw q agrees with direct physical reconstruction within FP32 roundoff.
            np.testing.assert_allclose(c-dc*h(phi)-cle,q,atol=5e-7,rtol=0)
            boundary={}
            for name,idx,targetp,targetconc,targetqbc in (
                ('pit_grid',np.flatnonzero(zero),0.,0.,-cle),
                ('solid_grid',np.flatnonzero(one),1.,1.,0.),
                ('pit_sample',np.arange(n,n+len(pit)),0.,0.,-cle),
                ('solid_sample',np.arange(n+len(pit),len(allpoints)),1.,1.,0.)):
                boundary[name]={'phi_error':stats(phi[:,idx]-targetp),
                                'c_error':stats(c[:,idx]-targetconc),
                                'q_error_over_c_le':stats((q[:,idx]-targetqbc)/cle),
                                'cL_over_c_le':stats(cl[:,idx]/cle)}
            dq=np.stack([matrix@v for v in q[:,:n]])
            targetdq=np.stack([matrix@v for v in targetq[[0,1,-1]]])
            pertime={}
            for it,(ts,jref) in enumerate(zip((0.,2.5,50.),(0,1,20))):
                pertime[str(ts)]={
                    'q_error_rms_over_c_le_free':rms((q[it,:n]-targetq[jref])[free])/cle,
                    'Dq_reference_rms_free':rms(targetdq[it,free]),
                    'Dq_prediction_rms_free':rms(dq[it,free]),
                    'Dq_error_relative_l2_free':float(np.linalg.norm((dq[it]-targetdq[it])[free])/np.linalg.norm(targetdq[it,free])),
                    'q_range_free':[float(q[it,:n][free].min()),float(q[it,:n][free].max())]}
            initial={'phi_error_rms_arc':rms((phi[0,:n]-targetphi[0])[arc]),
                     'c_error_rms_arc':rms((c[0,:n]-targetc[0])[arc]),
                     'q_error_rms_over_c_le_arc':rms((q[0,:n]-targetq[0])[arc])/cle,
                     'Dq_error_relative_l2_arc':float(np.linalg.norm((dq[0]-targetdq[0])[arc])/np.linalg.norm(targetdq[0,arc])),
                     'q_reference_range_arc':[float(targetq[0,arc].min()),float(targetq[0,arc].max())],
                     'q_prediction_range_arc':[float(q[0,:n][arc].min()),float(q[0,:n][arc].max())]}
            clamp={}
            for name,mask in (('pit_only',zero),('solid_only',one),('both',zero|one)):
                qc=q[0,:n].copy();qc[mask]=targetq[0,mask]
                action=matrix@qc
                clamp[name]={'change_Dq_rms_free':rms((action-dq[0])[free]),
                             'change_Dq_rms_initial_arc':rms((action-dq[0])[arc]),
                             'Dq_error_relative_l2_free':float(np.linalg.norm((action-targetdq[0])[free])/np.linalg.norm(targetdq[0,free])),
                             'Dq_error_relative_l2_initial_arc':float(np.linalg.norm((action-targetdq[0])[arc])/np.linalg.norm(targetdq[0,arc]))}
            records[str(step)]={'checkpoint':str(path),'boundary_all_three_times':boundary,
                               'physical_time_seconds':pertime,'initial_interface':initial,'frozen_q_boundary_clamp_at_t0':clamp}
            if step==5000:profiles[label]={'phi':phi[:,:n].reshape(3,*ref['X'].shape),
                                         'q':q[:,:n].reshape(3,*ref['X'].shape)}
        analytic_pit=initial_fields(cfg,np.asarray(pit)).astype(np.float64)
        data['runs'][label]={'path':str(run),'c_le':cle,'initial_arc_nodes':int(arc.sum()),
                            'analytic_ic_at_pit_samples_despite_zero_BC':{'phi':stats(analytic_pit[:,0]),'c':stats(analytic_pit[:,1])},
                            'checkpoints':records}
        b=records['5000']['boundary_all_three_times']
        print(label,'pit c max',b['pit_sample']['c_error']['max_abs'],'solid c max',b['solid_sample']['c_error']['max_abs'],
              'initial arc q RMSE/cle',records['5000']['initial_interface']['q_error_rms_over_c_le_arc'],flush=True)
    # Reference at final loop is identical across runs (validated by each loader).
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    ix=int(np.argmin(abs(ref['X'][0]-.5)));y=ref['Y'][:,ix]
    for row,(it,jref,title) in enumerate(((0,0,'0 s'),(2,20,'50 s'))):
        axes[row,0].plot(y,ref['phi'][jref,:,ix],'k--',label='reference')
        axes[row,1].plot(y,targetq[jref].reshape(ref['X'].shape)[:,ix]/cle,'k--',label='reference')
        for label,profile in profiles.items():
            axes[row,0].plot(y,profile['phi'][it,:,ix],label=label)
            axes[row,1].plot(y,profile['q'][it,:,ix]/cle,label=label)
        axes[row,0].set(title=f'{title}: phase',ylabel='phi',xlabel='y* at x*=0.5')
        axes[row,1].set(title=f'{title}: concentration drive',ylabel='q / c_le',xlabel='y* at x*=0.5')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    out=Path(__file__)
    out.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
    fig.savefig(out.with_suffix('.png'),dpi=150);plt.close(fig)

if __name__=='__main__':main()
