"""Independent CPU checkpoint replay and paired-architecture provenance checks.

Run from repository root with .venv, PYTHONPATH=., JAX_PLATFORMS=cpu.
All artifacts are explicit paths; no hashes or evolved training targets.
"""
from pathlib import Path
import json
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,load_bank,fields


root=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
runs=[root/name for name in ('20261029-confined-geometry-longer-direct-01',
                            '20261029-confined-geometry-longer-heterogeneous-01')]
configs=[ConfigDict(json.loads((run/'config.json').read_text())) for run in runs]
common=[]
for c in configs:
    record=json.loads(c.to_json());del record['arch'];record['log']['run_name']=''
    common.append(record)
assert common[0]==common[1], 'Nonarchitecture training/physical configuration differs'
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
manifest,refs,bank=load_bank(configs[0])
assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
draws=[];results=[]
for c,run in zip(configs,runs,strict=True):
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,jax.random.key(0)))
    predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c))
    maximum=0.
    for index,ref in enumerate(refs):
        data=jax.tree.map(lambda a:a[index],bank)
        actual=np.asarray(predict(model,data,jnp.asarray(ref['time'])))[:,:len(ref['points'])]
        with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as stored:
            saved=np.stack((stored['phi'],stored['mu']),axis=-1)
        maximum=max(maximum,float(np.max(abs(actual-saved))))
    assert maximum<1e-9,(run,maximum)
    with np.load(run/'training_draws.npz',allow_pickle=False) as f:
        ids=f['condition_ids'];draws.append(ids)
    assert ids.shape==(5000,4) and set(np.unique(ids))==set(range(16))
    summary=json.loads((run/'summary.json').read_text())
    metrics=json.loads((run/'evaluation/metrics.json').read_text())
    sensitivity=json.loads((run/'geometry_effect/metrics.json').read_text())
    for split in ('train','test'):
        for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
            np.testing.assert_allclose(summary['final_metrics'][split+'_'+key],metrics[split][key],atol=1e-14,rtol=0)
    results.append(dict(run=str(run),checkpoint=str(run/'model_final.eqx'),
        cpu_gpu_maximum_phi_mu_difference=maximum,native_test=metrics['test'],
        local_FEM=metrics['local_FEM'],geometry_response=sensitivity,
        wall_seconds=summary['wall_seconds'],warm_update_seconds=summary['mean_update_seconds_after_100'],
        memory=summary['device_memory_stats']))
np.testing.assert_array_equal(*draws)
result=dict(configuration_parity_except_architecture_and_run_name=True,
    architecture_scope='Bundle: input streams/positional lifting, depth, attention and width; no isolated causal attribution',
    actual_geometry_draws_match=draws[0].size,all16training_geometries_visited=True,
    no_heldout_geometries_in_updates=True,
    actual_time_draws_saved=False,time_parity_scope='Same runtime seed and unchanged time-sampling implementation; actual arrays not recorded',
    evolved_fields_absent_from_step_payload=True,
    heldout_scope='Eight reused development geometries; not fresh final generalization or seed-stability evidence',
    runs=results)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2),flush=True)
