"""Trained-checkpoint physical time derivative versus independent polynomial/Brent chain."""
from pathlib import Path
import sys,json,argparse
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.chebyshev import chebvander,chebder
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields,TanhPhaseNetwork
from examples.confined_ch2d.problem import load_reference,initial_field
from examples.confined_ch2d.operators import RegionalLaplacian


def audit(names):
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('CPU required')
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d');out=Path(__file__).with_suffix('.json')
    report=json.loads(out.read_text()) if out.exists() else dict(runs={})
    for name in names:
        run=base/name;c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
        v=ref['volumes'];x=jnp.asarray(ref['points']);vol=jnp.asarray(v);phi0=initial_field(ref['points'],c,xp=np);mean=float(v@phi0/v.sum())
        D=RegionalLaplacian(ref['points'],v,c);key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
        net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c,key))
        times=np.array([0.,.0001,.001,.005,.01,.03,.075,.15,.45,.9,1.5])
        @eqx.filter_jit
        def query(model):
            def at(t):return jax.jvp(lambda s:predict_fields(model,x,s,jnp.asarray(mean),vol,D)[:,0],(t,),(jnp.ones_like(t),))
            return jax.vmap(at)(jnp.asarray(times))
        actual,derivative=query(net);actual,derivative=np.asarray(actual),np.asarray(derivative)
        coefficients=np.asarray(eqx.filter_jit(jax.vmap(lambda p:net.net(net.inputs(p)[0])))(x))
        rank=c.arch.rank;L=np.log1p(c.physical.terminal_time/c.phase.initial_time_scale)
        fields=[];rates=[]
        for t in times:
            s=np.log1p(t/c.phase.initial_time_scale)/L;u=2*s-1
            B=chebvander(u,rank-1).ravel();Bd=(chebvander(u,rank-2)@chebder(np.eye(rank),axis=0)).ravel()
            increment=s*(coefficients@B);rate=coefficients@(B+2*s*Bd)/((t+c.phase.initial_time_scale)*L)
            if isinstance(net,TanhPhaseNetwork):
                a=net.tanh_amplitude;latent=np.arctanh(phi0/a)+increment;center=np.arctanh(mean/a)
                offset=brentq(lambda b:v@(a*np.tanh(latent+b))/v.sum()-mean,center-latent.max(),center-latent.min(),xtol=1e-14)
                phi=a*np.tanh(latent+offset);e=np.exp(-2*abs(latent+offset));J=a*4*e/(1+e)**2
                dt=J*(rate-(v*J)@rate/(v@J))
            else:
                phi=phi0+increment;phi+=mean-v@phi/v.sum();dt=rate-v@rate/v.sum()
            fields.append(phi);rates.append(dt)
        np.testing.assert_allclose(actual,fields,rtol=1e-9,atol=2e-9)
        np.testing.assert_allclose(derivative,rates,rtol=1e-8,atol=2e-8)
        report['runs'][name]=dict(phase_max_absolute_error=float(np.max(abs(actual-fields))),derivative_max_absolute_error=float(np.max(abs(derivative-rates))),times=times.tolist(),scope='Independent NumPy Chebyshev derivative and Brent mass shift, not AD finite differences only.')
        out.write_text(json.dumps(report,indent=2)+'\n');print(name,report['runs'][name],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='+',required=True);args=parser.parse_args();audit(args.names)
