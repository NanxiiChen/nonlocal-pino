"""Audit the new physical benchmark by explicit config/data/checkpoint paths.

Run from repository root in .venv with JAX_PLATFORMS=cpu and BLAS/OMP1.
"""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,load_bank,fields
from examples.confined_ch2d.operator_problem import evaluation_indices,fixed_initial


def audit(run,prior,output,preset):
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    saved=json.loads(c.to_json());saved['log']['run_name']='';del saved['train']['seed']
    assert saved==json.loads(importlib.import_module(preset).get_config().to_json())
    old=ConfigDict(json.loads((prior/'config.json').read_text()))
    for key in ('geometry','nl','arch','sample'):
        assert c[key].to_dict()==old[key].to_dict(),key
    training=c.train.to_dict();prior_training=old.train.to_dict()
    optimizer=training.pop('optimizer');prior_optimizer=prior_training.pop('optimizer')
    assert training==prior_training
    for key in ('beta','epsilon','mobility'):assert c.physical[key]==old.physical[key]
    assert c.physical.terminal_time==3. and old.physical.terminal_time==.75
    assert len(c.initial.waves)==10 and len(old.initial.waves)==8
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    manifest,refs,bank=load_bank(c);monitor,labeled=evaluation_indices(manifest)
    old_root=Path(old.operator.dataset_path);old_manifest=json.loads((old_root/'manifest.json').read_text())
    np.testing.assert_array_equal([r['holes'] for r in manifest['conditions']],[r['holes'] for r in old_manifest['conditions']])
    assert len(labeled)==24 and len(monitor)==16
    assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    for i in set(range(512))-set(monitor):assert 'phi' not in refs[i] and 'mu' not in refs[i]
    template=build_network(c,jax.random.key(0))
    a=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
    b=eqx.tree_deserialise_leaves(prior/'model_initial.eqx',template)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        if eqx.is_array(x):np.testing.assert_array_equal(x,y)
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
    predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c));maximum=0.
    for i in labeled:
        d=refs[i];item=jax.tree.map(lambda x:x[i],bank)
        pred=np.asarray(predict(model,item,jnp.asarray(d['time'])))[:,:len(d['points'])]
        with np.load(run/f'prediction_{i:03d}.npz',allow_pickle=False) as f:
            maximum=max(maximum,float(abs(pred-np.stack((f['phi'],f['mu']),axis=-1)).max()))
    assert maximum<1e-9,maximum
    with np.load(run/'training_draws.npz',allow_pickle=False) as f:ids=f['condition_ids']
    with np.load(prior/'training_draws.npz',allow_pickle=False) as f:old_ids=f['condition_ids']
    assert ids.shape==(20000,4);np.testing.assert_array_equal(ids,old_ids)
    assert set(np.unique(ids))==set(range(512))
    draw_map=np.asarray([r['draw_index'] for r in manifest['conditions']])
    assert not np.isin(draw_map[ids],c.operator.precompute.reserved_test_draws).any()
    local=json.loads((Path(c.operator.dataset_path)/'local_checks.json').read_text())
    assert len(local)==len(c.operator.local_validation_indices)==0
    assert manifest['local_reference_status']=='not_requested'
    evaluation=json.loads((run/'evaluation/metrics.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    for split in ('train','test'):
        for key in ('phi_pooled','phi_terminal','mu_pooled','mu_terminal'):
            np.testing.assert_allclose(evaluation[split][key],summary['final_metrics'][split+'_'+key],atol=1e-14,rtol=0)
    physical=[]
    for i in manifest['test_indices']:
        d=refs[i];v=d['volumes'];phi=d['phi'][-1]
        np.testing.assert_allclose(d['phi'][0],fixed_initial(d['points'],c,xp=np),atol=1e-14,rtol=0)
        physical.append(dict(index=i,pure_fraction=float(v@(abs(phi)>=.8)/v.sum()),mixed_fraction=float(v@(abs(phi)<.3)/v.sum())))
    wave=np.asarray(c.initial.waves);k2=np.sum((np.pi*wave[:,:2]/np.asarray(c.geometry.axes))**2,axis=1)
    counts=np.bincount(ids.ravel(),minlength=512)
    report=dict(run=str(run),prior_run=str(prior),config_matches_explicit_preset=True,
        unchanged_geometry_architecture_operator_optimizer=optimizer==prior_optimizer,
        unchanged_geometry_architecture_operator=True,optimizer=optimizer,prior_optimizer=prior_optimizer,
        expected_preset=preset,changed_physical_IC_and_horizon=True,
        comparison_scope='New physical benchmark; no isolated numerical-method accuracy gain claim',
        initial_parameters_identical=True,actual80000condition_draws_identical=True,
        no_heldout_geometry_draws=True,visits_min=int(counts.min()),visits_max=int(counts.max()),
        initial_modes=10,initial_analytic_bound=float(c.initial.amplitude*abs(wave[:,2]).sum()),
        shortest_mode_wavelength=float(2*np.pi/np.sqrt(k2.max())),mesh_size=c.geometry.mesh_size,
        local_linearized_modal_growth=(c.physical.mobility*k2*(c.physical.beta-c.physical.epsilon**2*k2)).tolist(),
        optimizer_payload_has_no_future_fields=True,cpu_gpu_maximum_phi_mu_difference=maximum,
        native_time_phi_max=max(r['native_time_errors']['phi'] for r in manifest['conditions'] if r['has_reference']),
        native_time_mu_max=max(r['native_time_errors']['mu'] for r in manifest['conditions'] if r['has_reference']),
        local_reference_checks=local,reference_terminal_phase_statistics=physical,
        native_test=evaluation['test'],local_FEM=evaluation['local_FEM'],
        wall_seconds=summary['wall_seconds'],warm_update_seconds=summary['mean_update_seconds_after_100'],
        memory=summary['device_memory_stats'],scope='One cold seed, finite512/8development cohort; native-only, local FEM deferred')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',default='examples.confined_ch2d.configs.train_operator_smallscale10_512')
    a=p.parse_args();audit(a.run,a.prior,a.output,a.config)
