"""Supervised representation DIAGNOSTIC ONLY; never a physics-trained result."""
from pathlib import Path
import sys,json,argparse,time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from examples.confined_ch2d.configs.train_coarsening import get_config as baseline
from examples.confined_ch2d.configs.train_coarsening_blend import get_config as blend
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.train import _build_network
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy,with_run_seed


def main(kind):
    c=with_run_seed((baseline if kind=='additive' else blend)(),42)
    configure_gpu_policy(c.train.gpu_policy);jax.config.update('jax_enable_x64',True)
    ref=load_reference(c);x=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes']);t=jnp.asarray(ref['time'])
    mean=jnp.dot(v,initial_field(x,c,xp=jnp))/v.sum()
    truth=jnp.stack((ref['phi_nonlocal'],ref['mu_nonlocal']),axis=-1)
    scale=jnp.sqrt(jnp.sum(truth**2*v[None,:,None],axis=(0,1))/(len(t)*v.sum()))
    key,_=jax.random.split(jax.random.PRNGKey(42));net=_build_network(c,key)
    opt=make_optimizer(c);state=opt.init(eqx.filter(net,eqx.is_array))
    @eqx.filter_jit
    def fields(model):return jax.vmap(lambda ti:predict_fields(model,x,ti,mean,v))(t)
    def loss(model):return jnp.sum(((fields(model)-truth)/scale)**2*v[None,:,None])/(v.sum()*len(t))
    @eqx.filter_jit
    def step(model,st):
        value,g=eqx.filter_value_and_grad(loss)(model)
        updates,st=opt.update(g,st,model)
        return eqx.apply_updates(model,updates),st,value
    run=Path(c.log.log_dir)/('20261027-chopt-representation-'+kind+'-01');run.mkdir(exist_ok=False)
    (run/'config.json').write_text(c.to_json(indent=2)+'\n');history=[];start=time.perf_counter()
    for i in range(1,3001):
        net,state,value=step(net,state)
        if i==1 or i%500==0:
            p=np.asarray(fields(net));row=dict(step=i,loss=float(value),elapsed=time.perf_counter()-start,
                phi=relative(p[...,0],ref['phi_nonlocal'],ref['volumes']),
                mu=relative(p[...,1],ref['mu_nonlocal'],ref['volumes']),
                phi_terminal=relative(p[-1,:,0],ref['phi_nonlocal'][-1],ref['volumes']))
            history.append(row);print(row,flush=True)
    eqx.tree_serialise_leaves(run/'supervised-diagnostic.eqx',net)
    result=dict(scope='SUPERVISED DIAGNOSTIC, not PINN. Shows demonstrated fit capacity only; no PDE accuracy claim.',kind=kind,run=str(run),history=history)
    (run/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (Path(__file__).parent/('chopt-representation-'+kind+'.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('additive','blend'),required=True)
    main(p.parse_args().kind)
