"""Independent lifted-metric objective, constant-mode share and dense-time check."""
from pathlib import Path
import sys,json,argparse
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields,TanhPhaseNetwork
from examples.confined_ch2d.tanh_mapping import phase_state
from examples.confined_ch2d.problem import load_reference,initial_field
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix


def audit(names,checkpoint):
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('Independent CPU audit required')
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d');out=Path(__file__).with_suffix('.json')
    report=json.loads(out.read_text()) if out.exists() else dict(scope='Independent physical residual/metric audit; no target fields train or reweight these models.',runs={})
    c0=ConfigDict(json.loads((base/names[0]/'config.json').read_text()));ref=load_reference(c0)
    v=ref['volumes'];root=np.sqrt(v);points=jnp.asarray(ref['points']);vol=jnp.asarray(v)
    mean=jnp.dot(vol,initial_field(points,c0,xp=jnp))/vol.sum();D=RegionalLaplacian(ref['points'],v,c0);A=native_matrix(ref['points'],v,c0)
    S=root[:,None]*A.toarray()/root[None,:];q=root/np.linalg.norm(root);factor=cho_factor(.5*(S+S.T)+np.outer(q,q),lower=True)
    @eqx.filter_jit
    def states(net,times):
        def at(t):
            values,dt=jax.jvp(lambda tt:predict_fields(net,points,tt,mean,vol,D),(t,),(jnp.ones_like(t),))
            J=phase_state(net,points,t,mean,vol)[1] if isinstance(net,TanhPhaseNetwork) else jnp.ones_like(values[:,0])
            return values,dt[:,0],J
        return jax.vmap(at)(times)
    for name in names:
        run=base/name;c=ConfigDict(json.loads((run/'config.json').read_text()))
        if c.loss.phase_metric.kind!='neumann_inverse':raise ValueError('Audit currently requires the explicit lifted Neumann metric')
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));model=eqx.tree_deserialise_leaves(run/checkpoint,_build_network(c,key))
        fit=np.r_[0.,c.sample.temporal.nodes];L=np.log1p(c.physical.terminal_time/c.phase.initial_time_scale)
        coords=np.log1p(fit/c.phase.initial_time_scale)/L;mid=.5*(coords[:-1]+coords[1:]);dense=np.sort(np.r_[fit,c.phase.initial_time_scale*np.expm1(L*mid)])
        mapped='residual_transform' in c.loss and c.loss.residual_transform.kind=='inverse_tanh_jacobian'
        rows={}
        for label,times in [('training_nodes',fit),('with_midpoints',dense)]:
            values,dt,J=states(model,jnp.asarray(times));values,dt,J=np.asarray(values),np.asarray(dt),np.asarray(J)
            residual=dt+c.physical.mobility*(A@values[...,1].T).T
            weighted=residual/J if mapped else residual
            inverse=cho_solve(factor,(weighted*root).T).T/root/c.physical.mobility
            raw_inverse=cho_solve(factor,(residual*root).T).T/root/c.physical.mobility
            per=np.sum(v*inverse**2,axis=1)/v.sum();constant=(weighted@v/v.sum()/c.physical.mobility)**2
            after=np.mean(np.sum(v*(raw_inverse/J)**2,axis=1)/v.sum())
            rows[label]=dict(count=len(times),loss=float(np.mean(per)),constant_mode_loss=float(np.mean(constant)),
                constant_mode_fraction=float(np.mean(constant)/np.mean(per)),
                before_metric_jacobian_loss=float(np.mean(np.sum(v*(cho_solve(factor,((residual/J)*root).T).T/root/c.physical.mobility)**2,axis=1)/v.sum())),
                after_metric_jacobian_loss=float(after),minimum_jacobian=float(J.min()),
                max_physical_residual_mean=float(np.max(abs(residual@v/v.sum()))),
                max_transformed_residual_mean=float(np.max(abs((residual/J)@v/v.sum()))),
                mass_max_drift=float(np.max(abs(values[...,0]@v/v.sum()-float(mean)))))
        report['runs'][name+'::'+checkpoint]=rows
        out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(name,checkpoint,rows,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='+',required=True);parser.add_argument('--checkpoint',default='model_final.eqx')
    args=parser.parse_args();audit(args.names,args.checkpoint)
