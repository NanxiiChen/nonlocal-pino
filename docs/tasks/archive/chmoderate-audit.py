"""Independent CPU checkpoint/physics audit of the intermediate CH experiment."""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.evaluate import field_plot


def audit(run):
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('CPU replay required')
    out=Path(__file__).parent;c=ConfigDict(json.loads((run/'config.json').read_text()))
    ref=load_reference(c);v=ref['volumes'];x=jnp.asarray(ref['points']);vol=jnp.asarray(v)
    mean=jnp.dot(vol,initial_field(x,c,xp=jnp))/vol.sum()
    D=RegionalLaplacian(ref['points'],v,c);A=native_matrix(ref['points'],v,c)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));template=_build_network(c,key)
    @eqx.filter_jit
    def query(net,times):
        return jax.vmap(lambda t:predict_fields(net,x,t,mean,vol,D))(times)
    dense_times=np.linspace(0,c.physical.terminal_time,151)
    energy=lambda p:np.sum(v*(.25*c.physical.beta*(p*p-1)**2+
        .5*c.physical.epsilon**2*p*(A@p.T).T),axis=1)
    report=dict(run=str(run),reference=c.reference.path,checkpoints={},
        reference_checks=json.loads(Path(c.reference.path).with_suffix('.json').read_text()))
    report['reference_morphology']=dict(
        relative_change=relative(ref['phi_nonlocal'][-1],ref['phi_nonlocal'][0],v),
        native_range=[float(ref['phi_nonlocal'].min()),float(ref['phi_nonlocal'].max())],
        initial_pure_fraction=float(v[abs(ref['phi_nonlocal'][0])>.9].sum()/v.sum()),
        terminal_pure_fraction=float(v[abs(ref['phi_nonlocal'][-1])>.9].sum()/v.sum()))
    frozen=initial_field(ref['points'],c,xp=np)
    report['frozen_initial_baseline']={}
    for target in ('local','nonlocal'):
        truth=ref['phi_'+target]
        report['frozen_initial_baseline'][target]=dict(
            pooled=relative(np.broadcast_to(frozen,truth.shape),truth,v),
            terminal=relative(frozen,truth[-1],v))
    for stage,stored in [('final','prediction.npz'),('best','prediction_best.npz')]:
        model=eqx.tree_deserialise_leaves(run/('model_'+stage+'.eqx'),template)
        actual=np.asarray(query(model,jnp.asarray(ref['time'])))
        with np.load(run/stored,allow_pickle=False) as data:
            replay={}
            for j,k in enumerate(('phi','mu')):
                np.testing.assert_allclose(actual[...,j],data[k],rtol=1e-8,atol=1e-8)
                replay[k]=float(np.max(abs(actual[...,j]-data[k])))
        phi,mu=actual[...,0],actual[...,1]
        chemical=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(A@phi.T).T
        np.testing.assert_allclose(mu,chemical,rtol=1e-9,atol=1e-9)
        dense=np.asarray(query(model,jnp.asarray(dense_times)))
        e=energy(dense[...,0])
        row=dict(cpu_replay_max_absolute=replay,
            chemical_max_defect=float(np.max(abs(mu-chemical))),
            mass_max_drift=float(np.max(abs(dense[...,0]@v/v.sum()-float(mean)))),
            phase_range=[float(dense[...,0].min()),float(dense[...,0].max())],
            terminal_mu_native_error_rms=float(np.sqrt(np.sum(v*(mu[-1]-ref['mu_nonlocal'][-1])**2)/v.sum())),
            terminal_mu_native_reference_rms=float(np.sqrt(np.sum(v*ref['mu_nonlocal'][-1]**2)/v.sum())),
            energy_max_increase=max(0.,float(np.max(np.diff(e)))),
            energy_time=dense_times.tolist(),energy=e.tolist(),errors={})
        for j,k in enumerate(('phi','mu')):
            row['errors'][k]={}
            for target in ('local','nonlocal'):
                truth=ref[k+'_'+target];a=actual[...,j]
                row['errors'][k][target]=dict(pooled=relative(a,truth,v),
                    positive=relative(a[1:],truth[1:],v),terminal=relative(a[-1],truth[-1],v),
                    per_time=[relative(aa,bb,v) for aa,bb in zip(a,truth,strict=True)])
        report['checkpoints'][stage]=row
        field_plot(ref,[ref['phi_local'],phi,abs(phi-ref['phi_local'])],
            out/('chmoderate-'+stage+'-fields.png'),['FEM reference','PINN phi','Absolute error'])
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for k in ('phi','mu'):
        axes[0].semilogy([r['step'] for r in history],
            [r['pooled_rel_l2_'+k+'_local'] for r in history],label=k+' pooled/FEM')
    axes[0].set(xlabel='Optimizer updates',ylabel='Relative L2');axes[0].legend();axes[0].grid(alpha=.3)
    axes[1].plot(ref['time'],ref['energy_nonlocal'],label='Native reference')
    for stage,row in report['checkpoints'].items():
        axes[1].plot(row['energy_time'],row['energy'],label='PINN '+stage)
    axes[1].set(xlabel='Time',ylabel='Native free energy');axes[1].legend();axes[1].grid(alpha=.3)
    fig.savefig(out/'chmoderate-convergence-energy.png',dpi=180);plt.close(fig)
    for filename in ('config.json','summary.json','best_metrics.json','metrics.jsonl','loss.jsonl','refinement_summary.json'):
        shutil.copyfile(run/filename,out/('chmoderate-'+filename))
    shutil.copyfile(Path(c.reference.path).with_suffix('.json'),out/'chmoderate-reference.json')
    (out/'chmoderate-analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in row.items() if kk not in ('energy_time','energy')}
        for k,row in report['checkpoints'].items()},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    audit(parser.parse_args().run)
