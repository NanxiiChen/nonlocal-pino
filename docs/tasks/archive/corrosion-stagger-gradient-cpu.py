import ast,json
from pathlib import Path
import equinox as eqx
import jax,jax.numpy as jnp,numpy as np
from ml_collections import ConfigDict
from examples.corrosion2d import residual_experiment as runner
from examples.corrosion2d import train as production
from examples.corrosion2d.residuals import make_loss_vector
from nonlocal_pinn.operators import build_D

root=Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-residual-phase-control-seed42')
c=ConfigDict(json.loads((root/'config.json').read_text()))
source=(root/'source/examples/corrosion2d/residual_experiment.py').read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_step')
namespace={'eqx':eqx,'jax':jax,'jnp':jnp}
exec(ast.get_source_segment(source,node),namespace)
class Recorder:
 def update(self,grad,state,params):return jax.tree.map(jnp.zeros_like,grad),grad
xx,yy,p=runner.grid(c);x=jnp.asarray(p,dtype=jnp.float32)
op=build_D(x,c.nl.delta,method=c.nl.method,kernel=c.nl.kernel)
pool=op.build_subgraph_pool(c.sample.K_core,c.sample.K_pool)
ids,sub,mask,_=pool[0]
zero,one,_=runner.boundary_values(c,p)
mask=mask & jnp.asarray(~(zero|one))[ids]
xbc0,xbc1=production._bc_info(c.pits.entries,c.sample.N_bc_dir,c.sample.N_bc_solid)
xic=production._sample_ic_points(c.pits.entries,c.sample.N_ic_global,c.sample.N_ic_local)
ic=jnp.asarray(runner.initial_fields(c,np.asarray(xic)),dtype=jnp.float32)
loss=make_loss_vector(c,runner.nondim(c),xic,ic,xbc0,xbc1)
kn,kt=jax.random.split(jax.random.PRNGKey(c.train.seed))
net=runner.build_model(c,kn)
net=eqx.tree_deserialise_leaves(root/'model_initial.eqx',net)
t=jax.random.uniform(jax.random.fold_in(kt,0),(c.sample.Nt,),minval=c.sample.t_min,maxval=1.)
old=namespace['make_step'](loss,Recorder(),c)
new=runner.make_step(loss,Recorder(),c)
a=old(net,None,jnp.ones(5),t,x[ids],sub,mask,jnp.array(True))
b=new(net,None,jnp.ones(5),t,x[ids],sub,mask,jnp.array(True),jnp.ones(5,dtype=bool))
va=np.concatenate([np.asarray(v,dtype=np.float64).ravel() for v in jax.tree.leaves(a[1]) if eqx.is_array(v)])
vb=np.concatenate([np.asarray(v,dtype=np.float64).ravel() for v in jax.tree.leaves(b[1]) if eqx.is_array(v)])
np.testing.assert_allclose(a[3],b[3],rtol=1e-5,atol=1e-7)
np.testing.assert_allclose(a[2],b[2],rtol=1e-5,atol=1e-7)
relative=float(np.linalg.norm(va-vb)/np.linalg.norm(va))
assert relative<1e-5,relative
print(json.dumps({'scope':'Actual production-sized first graph batch, real initialization, CPU FP32, recorded parameter gradients; old saved make_step versus new joint mask', 'gradient_relative_l2_difference':relative,'gradient_max_abs_difference':float(np.max(np.abs(va-vb))),'losses_old':np.asarray(a[3]).tolist(),'losses_new':np.asarray(b[3]).tolist(),'weights_old':np.asarray(a[2]).tolist(),'weights_new':np.asarray(b[2]).tolist()},indent=2))
