"""CPU replay, reference-identity and actual-draw audit of the512-domain run.

Run from root: PYTHONPATH=. JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 .venv/bin/python docs/tasks/archive/confined-richer512-audit.py
"""
import json
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,load_bank,fields
from examples.confined_ch2d.operator_problem import evaluation_indices
from examples.confined_ch2d.configs.train_operator_richer512 import get_config


run=Path('/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer512-01')
old_run=run.parent/'20261029-confined-geometry-richer-01'
c=ConfigDict(json.loads((run/'config.json').read_text()))
old=ConfigDict(json.loads((old_run/'config.json').read_text()))
current=json.loads(get_config().to_json());saved=json.loads(c.to_json())
del saved['train']['seed'];saved['log']['run_name']=''
assert current==saved
for key in ('geometry','physical','initial','nl','arch','sample','train'):
    assert c[key].to_dict()==old[key].to_dict(),key
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
manifest,refs,bank=load_bank(c);monitor,labeled=evaluation_indices(manifest)
assert manifest['pde_trajectory_solves']==0 and manifest['fem_solves']==0
assert len(monitor)==16 and len(labeled)==24
assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
assert all('phi' not in refs[i] and 'mu' not in refs[i] for i in set(range(512))-set(monitor))
template=build_network(c,jax.random.key(0))
new_initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
old_initial=eqx.tree_deserialise_leaves(old_run/'model_initial.eqx',template)
for a,b in zip(jax.tree.leaves(new_initial),jax.tree.leaves(old_initial),strict=True):
    if eqx.is_array(a):np.testing.assert_array_equal(a,b)
model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c))
maximum=0.
source=Path(c.operator.precompute.source_path)
old_manifest=json.loads((source/'manifest.json').read_text())
for index in labeled:
    d=refs[index];item=jax.tree.map(lambda a:a[index],bank)
    values=np.asarray(predict(model,item,jnp.asarray(d['time'])))[:,:len(d['points'])]
    with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
        maximum=max(maximum,float(np.max(abs(values-np.stack((f['phi'],f['mu']),axis=-1)))))
    record=manifest['conditions'][index];draw=record['draw_index']
    with np.load(source/old_manifest['conditions'][draw]['file'],allow_pickle=False) as f:
        for key in ('points','volumes','mesh_points','mesh_cells','holes','time','phi','mu','metric'):
            np.testing.assert_array_equal(d[key],f[key])
assert maximum<1e-9,maximum
with np.load(run/'training_draws.npz',allow_pickle=False) as f:
    ids=f['condition_ids'];assert ids.shape==(5000,4) and set(np.unique(ids))==set(range(512))
counts=np.bincount(ids.ravel(),minlength=512)
draws=np.asarray([r['draw_index'] for r in manifest['conditions']])
assert not np.isin(draws[ids],c.operator.precompute.reserved_test_draws).any()
summary=json.loads((run/'summary.json').read_text())
evaluation=json.loads((run/'evaluation/metrics.json').read_text())
for split in ('train','test'):
    for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
        np.testing.assert_allclose(summary['final_metrics'][split+'_'+key],evaluation[split][key],atol=1e-14,rtol=0)
before=json.loads((old_run/'evaluation/metrics.json').read_text())
report=dict(run=str(run),dataset=c.operator.dataset_path,
    same_physics_architecture_optimizer_budget_and_initial_parameters=True,
    cpu_gpu_maximum_phi_mu_difference=maximum,evaluation_conditions=labeled,
    analytic_initial_and_only_operator_arrays_in_optimizer_payload=True,
    new_training_operators_without_future_fields=496,actual_geometry_draws=int(ids.size),
    distinct_training_geometries=len(np.unique(ids)),visits_per_geometry_min=int(counts.min()),
    visits_per_geometry_max=int(counts.max()),old_validation_draws_excluded=True,
    reference_arrays_unchanged=True,
    precompute_seconds=manifest['seconds'],workers=manifest['workers'],bank_bytes=manifest['bytes'],
    nodes=[min(len(d['points']) for d in refs),max(len(d['points']) for d in refs)],
    old16_native_test=before['test'],new512_native_test=evaluation['test'],
    old16_local_FEM=before['local_FEM'],new512_local_FEM=evaluation['local_FEM'],
    wall_seconds=summary['wall_seconds'],warm_update_seconds=summary['mean_update_seconds_after_100'],
    memory=summary['device_memory_stats'],
    scope='Same8development validation geometries, single cold run, fixed5000/B4budget; no seed-stability/arbitrary-topology/continuum claim')
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
