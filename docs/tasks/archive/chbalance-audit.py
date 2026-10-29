"""CPU replay/physical diagnostics of explicitly named global PINN runs."""
from pathlib import Path
import sys,json,argparse,shutil
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix

BASE=Path('/root/autodl-tmp/tf-logs/confined_ch2d');OUT=Path(__file__).parent


def audit(names,append):
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('Independent CPU replay required')
    result_path=OUT/'chbalance-analysis.json'
    report=json.loads(result_path.read_text()) if append else dict(scope='Fresh shared two-head global fixed/gradient-norm/stagger controls; frozen physical reference, no target labels',runs={})
    c0=ConfigDict(json.loads((BASE/names[0]/'config.json').read_text()));ref=load_reference(c0)
    points=jnp.asarray(ref['points']);v=ref['volumes'];vol=jnp.asarray(v)
    D=RegionalLaplacian(ref['points'],v,c0);A=native_matrix(ref['points'],v,c0)
    mean=jnp.dot(vol,initial_field(points,c0,xp=jnp))/vol.sum()
    @eqx.filter_jit
    def query(net,times):return jax.vmap(lambda t:predict_fields(net,points,t,mean,vol,D))(times)
    @eqx.filter_jit
    def derivative(net,times):
        def at(t):return jax.jvp(lambda tt:predict_fields(net,points,tt,mean,vol,D),(t,),(jnp.ones_like(t),))[1][:,0]
        return jax.vmap(at)(times)
    diagnostic_times=np.array([0.,.0001,.001,.005,.01,.03,.075,.15,.45,.9,1.5])
    for name in names:
        run=BASE/name;c=ConfigDict(json.loads((run/'config.json').read_text()))
        # All controls belong to the same validated physical case/grid.
        actual_ref=load_reference(c)
        np.testing.assert_array_equal(actual_ref['points'],ref['points'])
        if str(actual_ref['identity'])!=str(ref['identity']):raise ValueError('Different physical identities in matched audit')
        np.testing.assert_array_equal(actual_ref['volumes'],ref['volumes'])
        np.testing.assert_array_equal(actual_ref['time'],ref['time'])
        if 'windows' in c.train:raise ValueError('Global audit rejects sequential window models')
        summary=json.loads((run/'summary.json').read_text());best=json.loads((run/'best_metrics.json').read_text())
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));template=_build_network(c,key)
        model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
        actual=np.asarray(query(model,jnp.asarray(ref['time'])))
        max_errors={}
        with np.load(run/'prediction.npz',allow_pickle=False) as stored:
            for j,field in enumerate(('phi','mu')):
                np.testing.assert_allclose(actual[...,j],stored[field],atol=1e-8,rtol=1e-8)
                max_errors[field]=float(np.max(abs(actual[...,j]-stored[field])))
        phi=actual[...,0];mu=actual[...,1]
        chemical=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(A@phi.T).T
        if c.formulation.chemical=='eliminated':np.testing.assert_allclose(mu,chemical,atol=1e-9,rtol=1e-9)
        dense_times=np.linspace(0,c.physical.terminal_time,151)
        dense=np.asarray(query(model,jnp.asarray(dense_times)))
        dense_phi=dense[...,0]
        native_energy=np.sum(v*(.25*c.physical.beta*(dense_phi**2-1)**2+.5*c.physical.epsilon**2*dense_phi*(A@dense_phi.T).T),axis=1)
        rms=lambda array:np.sqrt(np.sum(v*array*array,axis=-1)/v.sum())
        dt=np.asarray(derivative(model,jnp.asarray(diagnostic_times)))
        diagnostic=np.asarray(query(model,jnp.asarray(diagnostic_times)))
        flux=c.physical.mobility*(A@diagnostic[...,1].T).T;residual=dt+flux
        bestmodel=eqx.tree_deserialise_leaves(run/'model_best.eqx',template)
        bestfields=np.asarray(query(bestmodel,jnp.asarray(ref['time'])))
        with np.load(run/'prediction_best.npz',allow_pickle=False) as stored:
            for j,field in enumerate(('phi','mu')):np.testing.assert_allclose(bestfields[...,j],stored[field],atol=1e-8,rtol=1e-8)
        diagnostic_chemical=c.physical.beta*(diagnostic[...,0]**3-diagnostic[...,0])+c.physical.epsilon**2*(A@diagnostic[...,0].T).T
        row=dict(run=str(run),summary=summary,best=best,
            cpu_replay_max_absolute=max_errors,
            parameter_count=sum(int(a.size) for a in jax.tree.leaves(eqx.filter(model,eqx.is_inexact_array))),
            mass_max_drift=float(np.max(abs(phi@v/v.sum()-float(mean)))),
            energy_time=dense_times.tolist(),energy=native_energy.tolist(),energy_max_increase=max(0.,float(np.max(np.diff(native_energy)))),
            chemical_max_defect=float(np.max(abs(mu-chemical))),
            chemical_defect_rms=rms(mu-chemical).tolist(),
            terminal_pure_fraction=float(v[abs(phi[-1])>.9].sum()/v.sum()),
            phase_range=[float(dense_phi.min()),float(dense_phi.max())],
            positive_mu_native=relative(mu[1:],ref['mu_nonlocal'][1:],v),
            terminal_mu_native_error_rms=float(rms(mu[-1]-ref['mu_nonlocal'][-1])),
            terminal_mu_native_reference_rms=float(rms(ref['mu_nonlocal'][-1])),
            differential=dict(time=diagnostic_times.tolist(),residual_rms=rms(residual).tolist(),
                time_derivative_rms=rms(dt).tolist(),diffusive_term_rms=rms(flux).tolist(),
                energy_derivative=np.sum(v*diagnostic_chemical*dt,axis=1).tolist(),
                negative_dissipation=(-np.sum(v*diagnostic[...,1]*flux,axis=1)).tolist()))
        if 'causal' in c.loss:
            from scipy.linalg import cho_factor,cho_solve
            nodes=np.r_[0.,np.asarray(c.sample.temporal.nodes)];root=np.sqrt(v)
            S=(root[:,None]*A.toarray())/root[None,:];q=root/np.linalg.norm(root)
            factor=cho_factor(.5*(S+S.T)+np.outer(q,q),lower=True)
            weights_report={}
            for stage,candidate in [('soap',eqx.tree_deserialise_leaves(run/'model_soap.eqx',template)),('final',model)]:
                queried=np.asarray(query(candidate,jnp.asarray(nodes)))
                dphi=np.asarray(derivative(candidate,jnp.asarray(nodes)))
                residual=dphi+c.physical.mobility*(A@queried[...,1].T).T
                inverse=cho_solve(factor,(root*residual).T).T/root/c.physical.mobility
                per=np.sum(v*inverse**2,axis=1)/v.sum()
                weights=np.maximum(np.exp(-c.loss.causal.strength*np.r_[0.,np.cumsum(per[:-1])]),c.loss.causal.minimum)
                weights_report[stage]=dict(minimum=float(weights.min()),terminal=float(weights[-1]),
                    weights=weights.tolist(),unweighted_losses=per.tolist())
            row['causal_weight_diagnostics']=dict(times=nodes.tolist(),stages=weights_report,
                scope='SOAP uses adaptive weights; final weights are diagnostic only because L-BFGS is uniform')
        from scipy.linalg import cho_factor,cho_solve
        all_times=jnp.asarray(np.r_[0.,c.sample.temporal.nodes])
        all_fields=np.asarray(query(model,all_times));all_dt=np.asarray(derivative(model,all_times))
        r=all_dt+c.physical.mobility*(A@all_fields[...,1].T).T
        if 'evolution_scale' in c.loss:
            transformed=r/c.loss.evolution_scale
        else:
            root=np.sqrt(v);S=(root[:,None]*A.toarray())/root[None,:];q=root/np.linalg.norm(root)
            factor=cho_factor(.5*(S+S.T)+np.outer(q,q),lower=True)
            transformed=cho_solve(factor,(root*r).T).T/root/c.physical.mobility/c.loss.chemical_scale
        chem=(all_fields[...,1]-c.physical.beta*(all_fields[...,0]**3-all_fields[...,0])-
            c.physical.epsilon**2*(A@all_fields[...,0].T).T)/c.loss.chemical_scale
        independent_parts=np.asarray([np.mean(rms(transformed)**2),np.mean(rms(chem)**2)])
        balance=json.loads((run/'balancing_summary.json').read_text())
        independent_value=float(np.dot(independent_parts,balance['refinement_effective_weights']))
        np.testing.assert_allclose(independent_value,summary['refinement']['final_loss'],rtol=1e-6,atol=1e-8)
        row['independent_objective']=dict(parts=independent_parts.tolist(),value=independent_value,
            reported=summary['refinement']['final_loss'],balance=balance)
        logs=[json.loads(line) for line in (run/'balancing.jsonl').read_text().splitlines()]
        if c.train.loss_balance.kind=='gradient_norm':
            refreshed=[line for line in logs if line['refreshed']]
            for line in refreshed:
                measured=np.asarray(line['gradient_norms']);expected=np.clip(measured.mean()/measured,1e-6,10000.)
                np.testing.assert_allclose(line['weights'],expected,rtol=1e-12,atol=1e-12)
            row['adaptive_weight_history']=dict(refreshes=len(refreshed),
                minimum=np.min([line['weights'] for line in logs],axis=0).tolist(),
                maximum=np.max([line['weights'] for line in logs],axis=0).tolist(),
                weight_cap_hits=np.sum(np.asarray([line['weights'] for line in refreshed])>=10000.,axis=0).tolist(),
                norm_floor_hits=np.sum(np.asarray([line['gradient_norms'] for line in refreshed])<=1e-6,axis=0).tolist(),
                norm_ceiling_hits=np.sum(np.asarray([line['gradient_norms'] for line in refreshed])>=1e6,axis=0).tolist())
        expected_counts=dict(phase_only=2000,chemical_only=2000,joint=1000) if c.train.loss_balance.kind=='stagger' else dict(phase_only=0,chemical_only=0,joint=5000)
        if balance['soap_update_counts']!=expected_counts:raise ValueError('Unexpected stage exposure counts')
        row['best_errors']={}
        for index,k in enumerate(('phi','mu')):
            row['best_errors'][k]={target:dict(pooled=relative(bestfields[...,index],ref[k+'_'+target],v),
                terminal=relative(bestfields[-1,:,index],ref[k+'_'+target][-1],v)) for target in ('local','nonlocal')}
        report['runs'][name]=row
        np.savez_compressed(run/'cpu_audit_fields.npz',phi=phi,mu=mu,best_phi=bestfields[...,0],best_mu=bestfields[...,1])
        for filename in ('config.json','summary.json','best_metrics.json','metrics.jsonl','loss.jsonl','refinement_summary.json','balancing_summary.json','balancing.jsonl'):
            shutil.copyfile(run/filename,OUT/(name+'-'+filename))
        result_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(name,summary['final_metrics']['pooled_rel_l2_phi_local'],max_errors,flush=True)
    report['status']='CPU replays passed; field accuracy and energy require separate assessment';result_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--names',nargs='+',required=True);parser.add_argument('--append',action='store_true')
    args=parser.parse_args();audit(args.names,args.append)
