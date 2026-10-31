"""CPU replay and physical/reference/display checks for the richer CH pilot.

Run from root in .venv with PYTHONPATH=. JAX_PLATFORMS=cpu.
"""
import json
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,load_bank,fields
from examples.confined_ch2d.configs.train_operator_richer import get_config


run=Path('/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer-01')
old_run=run.parent/'20261029-confined-geometry-longer-heterogeneous-01'
c=ConfigDict(json.loads((run/'config.json').read_text()))
old=json.loads((old_run/'config.json').read_text())
assert c.physical.to_dict()==old['physical']
assert json.loads(c.arch.to_json())==old['arch']
current=json.loads(get_config().to_json());saved=json.loads(c.to_json())
del saved['train']['seed'];saved['log']['run_name']=''
assert current==saved
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
manifest,refs,bank=load_bank(c)
assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,jax.random.key(0)))
predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c))
maximum=0.
for index,d in enumerate(refs):
    item=jax.tree.map(lambda a:a[index],bank)
    values=np.asarray(predict(model,item,jnp.asarray(d['time'])))[:,:len(d['points'])]
    with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
        maximum=max(maximum,float(np.max(abs(values-np.stack((f['phi'],f['mu']),axis=-1)))))
assert maximum<1e-9,maximum
with np.load(run/'training_draws.npz',allow_pickle=False) as f:
    ids=f['condition_ids'];assert ids.shape==(5000,4) and set(np.unique(ids))==set(range(16))
summary=json.loads((run/'summary.json').read_text())
evaluation=json.loads((run/'evaluation/metrics.json').read_text())
for split in ('train','test'):
    for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
        np.testing.assert_allclose(summary['final_metrics'][split+'_'+key],evaluation[split][key],atol=1e-14,rtol=0)
waves=np.asarray(c.initial.waves);k2=np.sum((np.pi*waves[:,:2]/np.asarray(c.geometry.axes))**2,axis=1)
sensitivity=json.loads((run/'geometry_effect/metrics.json').read_text())
local=json.loads((Path(c.operator.dataset_path)/'local_checks.json').read_text())
assert len(local)==2 and all(row['status']=='passed' for row in local)
before=json.loads((old_run/'evaluation/metrics.json').read_text())
smooth=json.loads((old_run/'evaluation_smooth/metrics.json').read_text())
for key in ('conditions','train','test','local_FEM'):assert before[key]==smooth[key]
report=dict(run=str(run),dataset=c.operator.dataset_path,recipe_matches_explicit_config=True,
    unchanged_physical_parameters_and_architecture=True,
    comparison_scope='New physical case: IC, geometry prior and actual mesh differ; not an isolated method improvement',
    cpu_gpu_maximum_phi_mu_difference=maximum,actual_geometry_draws=ids.size,
    all16train_geometries_visited=True,no_heldout_draws=True,evolved_labels_absent_from_step_payload=True,
    nodes=[min(len(d['points']) for d in refs),max(len(d['points']) for d in refs)],
    native_refinement_phi_max=max(row['native_time_errors']['phi'] for row in manifest['conditions']),
    native_refinement_mu_max=max(row['native_time_errors']['mu'] for row in manifest['conditions']),
    initial_modes=waves.tolist(),initial_absolute_bound=float(c.initial.amplitude*np.sum(abs(waves[:,2]))),
    linearized_local_modal_growth=(c.physical.mobility*k2*(c.physical.beta-c.physical.epsilon**2*k2)).tolist(),
    old_plot_raw_metrics_unchanged=True,local_reference_checks=local,
    native_test=evaluation['test'],local_FEM=evaluation['local_FEM'],geometry_response=sensitivity,
    wall_seconds=summary['wall_seconds'],warm_update_seconds=summary['mean_update_seconds_after_100'],
    memory=summary['device_memory_stats'],
    generalization_scope='Finite16/8bank, single run; no arbitrary topology, continuum/seed-stability claim')
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
