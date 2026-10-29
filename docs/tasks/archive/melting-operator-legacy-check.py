"""Direct-source2Dlegacy parity and existing checkpoint template check."""
import importlib.util
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from examples.electropolish2d.operator_training import _build_network
from ml_collections import ConfigDict

run=Path('/root/autodl-tmp/tf-logs/electropolish2d/20261029-electro-operator-match-nonlocal-01')
source=run/'source/src/nonlocal_pinn/nn/conditional.py'
spec=importlib.util.spec_from_file_location('legacy_conditional',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
kwargs=dict(image_shape=(1,4,8),patch_shape=(2,2),extent=(1.,.5),width=8,
    encoder_depth=1,heads=2,mlp_ratio=2,decoder_depth=2,spatial_features=3,
    spatial_scale=1.,time_features=2,time_scale=1.,local_dim=2,out_dim=2,
    conditioning='cross_attention',norm_eps=1e-5,key=jax.random.key(8))
a=ConditionalFieldOperator(**kwargs);b=module.ConditionalFieldOperator(**kwargs)
count=0
for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
    if eqx.is_array(x):np.testing.assert_array_equal(x,y);count+=1
image=jnp.arange(32,dtype=jnp.float32).reshape(1,4,8)/32
points=jnp.array([[.2,.3],[.4,.1],[.8,.2]])
times=jnp.array([.1,.7]);local=jnp.ones((3,2))
x=np.asarray(eqx.filter_jit(a)(image,points,times,local))
y=np.asarray(eqx.filter_jit(b)(image,points,times,local))
np.testing.assert_array_equal(x,y)
c=ConfigDict(json.loads((run/'config.json').read_text()))
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c.arch,c.operator.image_shape,jax.random.key(0)))
assert all(np.isfinite(np.asarray(z)).all() for z in jax.tree.leaves(net) if eqx.is_array(z))
result=dict(equal_initial_arrays=count,maximum_raw_output_gap=float(abs(x-y).max()),
    existing2Dcheckpoint_load=True,legacy_source=str(source),checkpoint=str(run/'model_final.eqx'))
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',result)
