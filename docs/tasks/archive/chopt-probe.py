"""Read-only IC-lifting and PDE parameter-gradient diagnostics; CPU FP64."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference,initial_field,boundary_samples,relative
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix


def main():
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('Run with JAX_PLATFORMS=cpu')
    run=Path('/root/autodl-tmp/tf-logs/confined_ch2d/20261027-coarsening10-nonlocal-01')
    c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
    x=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes'])
    p0=initial_field(ref['points'],c,xp=np);mean=jnp.dot(v,jnp.asarray(p0))/v.sum()
    A=native_matrix(ref['points'],ref['volumes'],c)
    mu0=c.physical.beta*(p0**3-p0)+c.physical.epsilon**2*(A@p0)
    velocity=-c.physical.mobility*(A@mu0)
    norm=lambda a:float(np.sqrt(np.sum(np.asarray(a)**2*ref['volumes'])))
    result=dict(scope='Reference data used only to diagnose required output coordinates, never PDE training labels.',
        initial_pde_time_scale=norm(p0-float(mean))/norm(velocity),lifting_targets={},gradient_probes={})
    for kind,tau in [('additive',c.physical.terminal_time),('blend015',.15),('blend005',.05)]:
        t=ref['time'][1:];truth=ref['phi_nonlocal'][1:]
        if kind=='additive':target=(truth-p0)/(t[:,None]/c.physical.terminal_time)
        else:
            g=np.exp(-t/tau);target=(truth-g[:,None]*p0)/(-np.expm1(-t/tau))[:,None]
        result['lifting_targets'][kind]=dict(
            first_time=float(t[0]),first_rms=norm(target[0])/np.sqrt(float(v.sum())),
            terminal_rms=norm(target[-1])/np.sqrt(float(v.sum())),
            first_roughness=float(target[0]@(ref['volumes']*(A@target[0]))),
            terminal_roughness=float(target[-1]@(ref['volumes']*(A@target[-1]))))
    b,n,bw,_=boundary_samples(c);D=RegionalLaplacian(ref['points'],ref['volumes'],c)
    times=(jnp.arange(8,dtype=jnp.float64)+.5)*c.physical.terminal_time/8
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));template=_build_network(c,key)
    models={name:eqx.tree_deserialise_leaves(run/file,template) for name,file in [('initial','model_initial.eqx'),('final','model_final.eqx')]}
    @eqx.filter_jit
    def replay(model):return jax.vmap(lambda t:predict_fields(model,x,t,mean,v))(jnp.asarray(ref['time']))
    with np.load(run/'prediction.npz') as saved:
        actual=np.asarray(replay(models['final']))
        for i,k in enumerate(('phi','mu')):np.testing.assert_allclose(actual[...,i],saved[k],atol=1e-8,rtol=1e-8)
        result['legacy_final_replay_max_absolute']=float(max(np.max(abs(actual[...,i]-saved[k])) for i,k in enumerate(('phi','mu'))))
    for include in (True,False):
        cfg=ConfigDict(c.to_dict());cfg.sample.include_initial_time=include
        loss=make_loss_fn(x,v,D,jnp.asarray(b),jnp.asarray(n),jnp.asarray(bw),cfg)
        @eqx.filter_jit
        def probe(model):
            def term(net,i):return loss(net,times,mean)[1][i]
            a,ga=eqx.filter_value_and_grad(lambda net:term(net,0))(model)
            d,gd=eqx.filter_value_and_grad(lambda net:term(net,1))(model)
            aa=jax.tree.leaves(eqx.filter(ga,eqx.is_array));dd=jax.tree.leaves(eqx.filter(gd,eqx.is_array))
            na=sum(jnp.vdot(q,q).real for q in aa);nd=sum(jnp.vdot(q,q).real for q in dd)
            dot=sum(jnp.vdot(q,r).real for q,r in zip(aa,dd,strict=True))
            return jnp.array([a,d,jnp.sqrt(na),jnp.sqrt(nd),dot/jnp.sqrt(na*nd)])
        for name,model in models.items():
            values=np.asarray(probe(model));label=name+('_with_t0' if include else '_positive')
            result['gradient_probes'][label]=dict(zip(('phase_loss','chemical_loss','phase_gradient_norm','chemical_gradient_norm','gradient_cosine'),map(float,values),strict=True))
            print(label,result['gradient_probes'][label],flush=True)
    path=Path(__file__).with_suffix('.json');path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
