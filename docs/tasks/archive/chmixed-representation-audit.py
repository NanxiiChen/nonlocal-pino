"""CPU replay and physical residuals of the supervised-only capacity diagnostic."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
import equinox as eqx
import jax,jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference,initial_field
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.losses import make_loss_fn
jax.config.update('jax_enable_x64',True)
if jax.default_backend()!='cpu':raise ValueError('CPU required')
run=Path('/root/autodl-tmp/tf-logs/confined_ch2d/20261028-chmixed-representation-01')
c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
x=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes']);mean=jnp.dot(v,initial_field(x,c,xp=jnp))/v.sum()
key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));model=eqx.tree_deserialise_leaves(run/'supervised_model.eqx',_build_network(c,key))
D=RegionalLaplacian(ref['points'],ref['volumes'],c);A=native_matrix(ref['points'],ref['volumes'],c)
@eqx.filter_jit
def query(net):return jax.vmap(lambda t:predict_fields(net,x,t,mean,v,D))(jnp.asarray(ref['time']))
actual=np.asarray(query(model));rms=lambda a:np.sqrt(np.sum(ref['volumes']*a*a,axis=-1)/ref['volumes'].sum())
with np.load(run/'supervised_prediction.npz') as f:
    replay={name:float(np.max(abs(actual[...,i]-f[name]))) for i,name in enumerate(('phi','mu'))}
    for i,name in enumerate(('phi','mu')):np.testing.assert_allclose(actual[...,i],f[name],rtol=1e-8,atol=1e-8)
phi,mu=actual[...,0],actual[...,1];chem=mu-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*(A@phi.T).T
report=dict(scope='Supervised-only diagnostic, not a PINN solution. Fields fit does not automatically establish a jointly accurate PDE trial space.',cpu_replay=replay,chemical_defect_rms=rms(chem).tolist(),physics_objectives={})
for name in ('neumann_inverse','mixed_block'):
    config=ConfigDict(c.to_dict());config.loss.phase_metric=ConfigDict(dict(kind=name))
    if name=='mixed_block':config.loss.phase_metric.time_scale=config.phase.initial_time_scale
    loss=make_loss_fn(x,v,D,None,None,None,config)
    value,parts=eqx.filter_jit(loss)(model,jnp.asarray(c.sample.temporal.nodes),mean)
    report['physics_objectives'][name]=dict(total=float(value),parts=np.asarray(parts).tolist());print(name,report['physics_objectives'][name],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
