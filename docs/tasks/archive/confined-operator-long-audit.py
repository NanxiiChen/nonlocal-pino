"""Audit a matched short/long cold training pair using explicit run paths.

Run in .venv with PYTHONPATH=. JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,load_bank,fields
from examples.confined_ch2d.operator_problem import evaluation_indices


def parameter_difference(first,second):
    maximum=0.;squared=0.;norm=0.
    for a,b in zip(jax.tree.leaves(first),jax.tree.leaves(second),strict=True):
        if eqx.is_array(a):
            a,b=np.asarray(a),np.asarray(b)
            maximum=max(maximum,float(abs(a-b).max()))
            squared+=float(np.sum((a-b)**2));norm+=float(np.sum(b*b))
    return dict(maximum_absolute=maximum,relative_l2=float(np.sqrt(squared/norm)))


def audit(run,short_run,output):
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    old=ConfigDict(json.loads((short_run/'config.json').read_text()))
    actual=json.loads(c.to_json());expected=json.loads(old.to_json())
    actual['log']['run_name']=expected['log']['run_name']
    assert actual['train']['iterations']==20000 and expected['train']['iterations']==5000
    actual['train']['iterations']=5000
    assert actual==expected,'Only iteration budget/run name should differ'
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    template=build_network(c,jax.random.key(0))
    initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
    old_initial=eqx.tree_deserialise_leaves(short_run/'model_initial.eqx',template)
    initial_difference=parameter_difference(initial,old_initial)
    assert initial_difference['maximum_absolute']==0.
    prefix=eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',template)
    old_final=eqx.tree_deserialise_leaves(short_run/'model_final.eqx',template)
    prefix_difference=parameter_difference(prefix,old_final)
    with np.load(run/'training_draws.npz',allow_pickle=False) as f:draws=f['condition_ids']
    with np.load(short_run/'training_draws.npz',allow_pickle=False) as f:old_draws=f['condition_ids']
    assert draws.shape==(20000,4) and old_draws.shape==(5000,4)
    np.testing.assert_array_equal(draws[:5000],old_draws)
    assert set(np.unique(draws))==set(range(512))
    manifest,refs,bank=load_bank(c);monitor,labeled=evaluation_indices(manifest)
    assert len(monitor)==16 and len(labeled)==24
    assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    assert all('phi' not in refs[i] and 'mu' not in refs[i] for i in set(range(512))-set(monitor))
    source_draws=np.asarray([r['draw_index'] for r in manifest['conditions']])
    assert not np.isin(source_draws[draws],c.operator.precompute.reserved_test_draws).any()
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
    predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c))
    maximum=0.
    for index in labeled:
        d=refs[index];item=jax.tree.map(lambda a:a[index],bank)
        values=np.asarray(predict(model,item,jnp.asarray(d['time'])))[:,:len(d['points'])]
        with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
            maximum=max(maximum,float(abs(values-np.stack((f['phi'],f['mu']),axis=-1)).max()))
    assert maximum<1e-9,maximum
    evaluation=json.loads((run/'evaluation/metrics.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    for split in ('train','test'):
        for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
            np.testing.assert_allclose(summary['final_metrics'][split+'_'+key],evaluation[split][key],rtol=0,atol=1e-14)
    repo=Path(__file__).resolve().parents[2]
    # This script may be archived one directory deeper after completion.
    if repo.name=='docs':repo=repo.parent
    for rel in ('examples/confined_ch2d/operator_training.py','examples/confined_ch2d/operator_problem.py',
                'examples/confined_ch2d/operators.py','src/nonlocal_pinn/nn/point_conditional.py',
                'src/nonlocal_pinn/training/optimizers.py'):
        assert (repo/rel).read_bytes()==(run/'source'/rel).read_bytes(),rel
    counts=np.bincount(draws.ravel(),minlength=512)
    report=dict(run=str(run),short_run=str(short_run),config_only_budget_and_run_name_differ=True,
        initial_parameter_difference=initial_difference,prefix5000_parameter_difference=prefix_difference,
        actual_condition_prefix5000_identical=True,actual_draws=int(draws.size),distinct_training_inputs=512,
        visits_min=int(counts.min()),visits_max=int(counts.max()),no_validation_overlap=True,
        optimizer_payload_has_no_future_fields=True,cpu_gpu_maximum_phi_mu_difference=maximum,
        evaluation_reference='Unchanged native targets; same refined local FEM on2domains',
        native_test=evaluation['test'],local_FEM=evaluation['local_FEM'],
        wall_seconds=summary['wall_seconds'],warm_update_seconds=summary['mean_update_seconds_after_100'],
        memory=summary['device_memory_stats'],
        scope='Same8development geometries and one cold seed; fixed20000step final, no validation-based checkpoint selection')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--short-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.run,args.short_run,args.output)
