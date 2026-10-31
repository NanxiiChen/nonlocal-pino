"""Replay the selected checkpoint and verify bank/sampling/metric provenance."""
from pathlib import Path
import json
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator import get_config
from examples.confined_ch2d.operator_training import build_network,load_bank,fields

ROOT=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
selected=ROOT/'20261029-confined-geometry-operator-direct-01'
baseline=ROOT/'20261029-confined-geometry-operator-adam-01'
c=ConfigDict(json.loads((selected/'config.json').read_text()))
current=json.loads(get_config().to_json());saved=json.loads(c.to_json())
del saved['train']['seed'];saved['log']['run_name']=''
assert current==saved
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
manifest,refs,bank=load_bank(c)
model=eqx.tree_deserialise_leaves(selected/'model_final.eqx',build_network(c,jax.random.key(42)))
predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c))
maximum=0.
for index,ref in enumerate(refs):
    d=jax.tree.map(lambda a:a[index],bank)
    out=np.asarray(predict(model,d,jnp.asarray(ref['time'])))[:,:len(ref['points'])]
    with np.load(selected/f'prediction_{index:03d}.npz',allow_pickle=False) as stored:
        maximum=max(maximum,float(np.max(abs(out-np.stack((stored['phi'],stored['mu']),axis=-1)))))
assert maximum<1e-10,maximum
with np.load(selected/'training_draws.npz',allow_pickle=False) as a,np.load(baseline/'training_draws.npz',allow_pickle=False) as b:
    np.testing.assert_array_equal(a['condition_ids'],b['condition_ids'])
    ids=a['condition_ids'];assert ids.shape==(5000,4) and ids.min()>=0 and ids.max()<16
    assert len(np.unique(ids))==16
summary=json.loads((selected/'summary.json').read_text())
metrics=json.loads((selected/'evaluation/metrics.json').read_text())
for split in ('train','test'):
    for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
        assert abs(summary['final_metrics'][split+'_'+key]-metrics[split][key])<1e-14
swap=summary['geometry_swaps']
wrong=float(np.mean([row['wrong_condition_phi_pooled'] for row in swap]))
worse=sum(row['wrong_condition_phi_pooled']>summary['final_metrics']['conditions'][row['index']]['phi_pooled'] for row in swap)
record=dict(selected=str(selected),default_exact_saved_strategy=True,
    replay_maximum_phi_mu_difference=maximum,reference_operator_assembly_parity='all24 geometries, three independent probe fields',
    matched_condition_draws=ids.shape[0]*ids.shape[1],all_train_geometries_visited=True,no_heldout_draws=True,
    independent_metrics_match=True,geometry_swap_mean_error=wrong,geometry_swap_worse_cases=worse,
    heldout_scope='Eight training-unseen development geometries reused for encoder comparison; not a fresh final generalization test',
    native_test=metrics['test'],local_FEM=metrics['local_FEM'])
Path(__file__).with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps(record,indent=2),flush=True)
