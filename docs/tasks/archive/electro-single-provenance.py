"""Actual condition-use and matched fixed-input initialization audit; no hashes."""
import ast,json
from pathlib import Path
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict
from examples.electropolish2d.operator_training import _build_network

root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
names=['20261028-electro-operator-single-cross-01','20261028-electro-operator-single-soap-01']
configs=[ConfigDict(json.loads((root/n/'config.json').read_text())) for n in names]
models=[eqx.tree_deserialise_leaves(root/n/'model_initial.eqx',_build_network(c.arch,c.operator.image_shape,jax.random.key(0))) for n,c in zip(names,configs)]
leaves=0
for a,b in zip(jax.tree.leaves(models[0]),jax.tree.leaves(models[1])):
 if eqx.is_array(a):np.testing.assert_array_equal(a,b);leaves+=1
arrays=[]
for name in names:
 used=np.load(root/name/'sampled_conditions.npy')
 assert used.shape==(2000,1);np.testing.assert_array_equal(used,0)
 arrays.append(dict(run=name,shape=list(used.shape),unique_indices=np.unique(used).tolist()))
a,b=[c.to_dict() for c in configs]
for d in (a,b):d['train']['optimizer']='CONTROL';d['log']['run_name']='CONTROL'
assert a==b
for filename in ['src/nonlocal_pinn/nn/conditional.py','src/nonlocal_pinn/operators/__init__.py']:
 assert (root/names[0]/'source'/filename).read_bytes()==(root/names[1]/'source'/filename).read_bytes()
for name in ('physical_fields','make_loss_vector','_build_network'):
 segments=[]
 for run in names:
  source=(root/run/'source/examples/electropolish2d/operator_training.py').read_text()
  tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  segments.append(ast.get_source_segment(source,node))
 assert segments[0]==segments[1],name
results=dict(initial_parameter_array_leaves_identical=leaves,actual_sample_arrays=arrays,
 config_difference_only_optimizer_and_run_name=True,model_operator_and_physical_loss_sources_identical=True,
 same_interface_index=0,surface_identity=1662057944,budget=2000,
 interpretation='fixed-input optimizer comparison only; one seed, no general optimizer ranking')
Path('docs/tasks/archive/electro-single-provenance.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
