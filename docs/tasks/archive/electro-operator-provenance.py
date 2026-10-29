"""Exact array/source comparisons, no hashes or sibling dependencies."""
import json
from pathlib import Path
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict
from examples.electropolish2d.operator_training import _build_network

root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
runs=[root/f'20261028-electro-operator-{k}-01' for k in ('cross','film')]
configs=[ConfigDict(json.loads((r/'config.json').read_text())) for r in runs]
models=[eqx.tree_deserialise_leaves(r/'model_initial.eqx',_build_network(c.operator,c.operator.image_shape,jax.random.key(0))) for r,c in zip(runs,configs)]
for c in configs:
    c.operator.conditioning='CONTROL';c.log.run_name='CONTROL'
assert configs[0].to_dict()==configs[1].to_dict()
common=['patch_projection','encoder','encoder_norm','spatial_projection','spatial_frequencies','time_film','layers','output']
count=0
for name in common:
    for a,b in zip(jax.tree.leaves(getattr(models[0],name)),jax.tree.leaves(getattr(models[1],name))):
        if eqx.is_array(a):np.testing.assert_array_equal(a,b);count+=1
files=json.loads((runs[0]/'source/files.json').read_text())
diffs=[f for f in files if (runs[0]/'source'/f).read_bytes()!=(runs[1]/'source'/f).read_bytes()]
assert diffs==['examples/electropolish2d/train.py'],diffs
# Only the CLI return to absl changed; training/architecture/objective stayed exact.
a=(runs[0]/'source'/diffs[0]).read_text();b=(runs[1]/'source'/diffs[0]).read_text()
assert a.replace('        return train_operator(with_run_seed(config, _SEED.value))','        train_operator(with_run_seed(config, _SEED.value))\n        return')==b
record=dict(common_initial_array_leaves=count,common_modules=common,
    configs_equal_except_conditioning_and_run_name=True,source_file_count=len(files),
    source_differences=diffs,source_difference_scope='absl success return only; no numerical change',
    both_dataset_paths=[c.operator.dataset_path for c in configs],
    training_protocol='same condition/time RNG and padded core-pool schedule; independent model optimization')
Path('docs/tasks/archive/electro-operator-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
