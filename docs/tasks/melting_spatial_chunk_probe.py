import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.icemelting3d.operator_data import build_network,physics
from examples.icemelting3d.ellipsoids import image_points,signed_distance
from examples.icemelting3d.operator_training import make_predictor
from examples.icemelting3d.operator_losses import latent_fields
configure_gpu_policy('ordinary');jax.config.update('jax_default_matmul_precision','highest')
run=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261031-melting-spatial-01-fixed_pool')
c=ConfigDict(json.loads((run/'config.json').read_text()))
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,jax.random.key(1)))
m=json.loads((Path(c.operator.dataset_path)/'manifest.json').read_text());params=jnp.asarray([m['parameters'][i] for i in [2,5]],dtype=jnp.float32)
width=physics(c)[3];a=np.linspace(0,1,128,dtype=np.float32)
x=jnp.asarray(np.stack(np.meshgrid(a,a[63:67],a,indexing='xy'),-1).reshape(-1,3))
s=jnp.asarray(image_points(tuple(c.arch.image_shape)),dtype=jnp.float32)
d=eqx.filter_jit(lambda x:jax.vmap(lambda p:signed_distance(x,p,40))(params)/width)
d0=eqx.filter_jit(lambda x,p:signed_distance(x,p,40)/width)
ims=jnp.tanh(d(s)).reshape((2,)+tuple(c.arch.image_shape))
im0=jnp.tanh(d0(s,params[0])).reshape(tuple(c.arch.image_shape));u0=d(x)[0];single=d0(x,params[0]);times=jnp.array([.25,.5,.75,1.],dtype=jnp.float32)
idx=np.random.default_rng(12).choice(len(x),512,replace=False)
expected=make_predictor(c)(net,np.asarray(params[0]),np.asarray(x)[idx],np.asarray(times))
encoded=eqx.filter_jit(lambda n,im:n.encode(im))(net,im0)
@eqx.filter_jit
def explicit(n,e,x,u0,t):
 prep=n.prepare_encoded(e,x,jnp.zeros((len(x),0),dtype=x.dtype))
 return jnp.tanh(u0[None]+t[:,None]*n.decode(prep,t)[...,0])
@eqx.filter_jit
def explicit_jvp(n,e,x,u0,t):
 return jnp.tanh(latent_fields(n,e,x,u0,t)[0])
@eqx.filter_jit
def dynamic(n,im,x,u0,t):
 return jnp.tanh(latent_fields(n,n.encode(im),x,u0,t)[0])
@eqx.filter_jit
def captured(n,x,u0):
 return jnp.tanh(latent_fields(n,n.encode(ims[0]),x,u0,times)[0])
r={'image_gap':float(jnp.max(abs(im0-ims[0]))),'initial_latent_gap':float(jnp.max(abs(u0-single)))}
for name,out in [('explicit',explicit(net,encoded,x,single,times)),('explicit_jvp',explicit_jvp(net,encoded,x,single,times)),('dynamic',dynamic(net,im0,x,single,times)),('captured',captured(net,x,u0))]:
 out=np.asarray(out);r[name+'_gap']=float(np.max(abs(out[:,idx]-expected)))
 print(name,r[name+'_gap'],flush=True)
Path('/root/nonlocal-pinn/docs/tasks/melting-spatial-chunk-probe.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
