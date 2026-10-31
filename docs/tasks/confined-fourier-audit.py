"""Audit a Fourier/capacity candidate or continuation on the same native CH bank.

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
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled,initial_operator_fields


def audit(run,prior,output,preset,bank_growth=False):
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    saved=json.loads(c.to_json());saved['log']['run_name']='';del saved['train']['seed']
    assert saved==json.loads(importlib.import_module(preset).get_config().to_json())
    old=ConfigDict(json.loads((prior/'config.json').read_text()))
    summary=json.loads((run/'summary.json').read_text())
    start_step=summary['start_step'] if 'start_step' in summary else 0
    for key in ('geometry','physical','initial','nl'):
        assert c[key].to_dict()==old[key].to_dict(),key
    for key in (set(c.loss)|set(old.loss))-{'initial_residual_fraction'}:
        assert c.loss[key]==old.loss[key],key
    initial_fraction=c.loss.initial_residual_fraction if 'initial_residual_fraction' in c.loss else 1/(c.sample.Nt+1)
    old_fraction=old.loss.initial_residual_fraction if 'initial_residual_fraction' in old.loss else 1/(old.sample.Nt+1)
    np.testing.assert_allclose(initial_fraction,old_fraction,rtol=0,atol=1e-15)
    operator_keys=set(c.operator)|set(old.operator)
    allowed_operator_changes={'metric','metric_workers','condition_batch','initial_rhs_condition'}
    if bank_growth:allowed_operator_changes|={'train_conditions','dataset_path','precompute'}
    for key in operator_keys-allowed_operator_changes:
        assert c.operator[key]==old.operator[key],key
    metric_changes={key:dict(before=old.operator[key] if key in old.operator else None,after=c.operator[key])
        for key in ('metric','metric_workers') if key in c.operator and (key not in old.operator or c.operator[key]!=old.operator[key])}
    architecture_changes={key:dict(before=old.arch[key] if key in old.arch else None,after=c.arch[key])
        for key in c.arch if key not in old.arch or c.arch[key]!=old.arch[key]}
    sampling_changes={key:dict(before=old.sample[key] if key in old.sample else None,after=c.sample[key])
        for key in c.sample if key not in old.sample or c.sample[key]!=old.sample[key]}
    training_changes={key:dict(before=old.train[key] if key in old.train else None,after=c.train[key])
        for key in c.train if key not in old.train or c.train[key]!=old.train[key]}
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    manifest,refs,bank=load_bank(c);monitor,labeled=evaluation_indices(manifest)
    old_root=Path(old.operator.dataset_path);old_manifest=json.loads((old_root/'manifest.json').read_text())
    growth=None
    if bank_growth:
        from docs.tasks.confined_bank_growth import verify_growth
        growth=verify_growth(Path(c.operator.dataset_path),old_root)
        for key in set(c.operator.precompute)|set(old.operator.precompute):
            if key not in ('source_path','reuse_mode'):assert c.operator.precompute[key]==old.operator.precompute[key],key
    else:
        np.testing.assert_array_equal([r['holes'] for r in manifest['conditions']],[r['holes'] for r in old_manifest['conditions']])
    assert len(labeled)==24 and len(monitor)==16
    expected_payload={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    if initial_rhs_enabled(c):expected_payload.add('domain_fields')
    assert set(bank)==expected_payload
    if initial_rhs_enabled(c):
        for i,d in enumerate(refs):
            known=initial_operator_fields(d,c);n=len(known)
            np.testing.assert_array_equal(np.asarray(bank['domain_fields'][i,:n]),known)
            padding=np.asarray(bank['domain_fields'][i,n:])
            np.testing.assert_array_equal(padding,np.zeros_like(padding))
    for i in set(range(c.operator.train_conditions))-set(monitor):assert 'phi' not in refs[i] and 'mu' not in refs[i]
    template=build_network(c,jax.random.key(0))
    a=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
    if start_step:
        from examples.confined_ch2d.operator_continuation import continuation_spec
        continuation_spec(c)
        expected=eqx.tree_deserialise_leaves(Path(c.train.continuation.parent_run)/'model_final.eqx',template)
    else:expected=build_network(c,jax.random.split(jax.random.key(c.train.seed),3)[0])
    initial_gap=0.
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(expected),strict=True):
        if eqx.is_array(x):initial_gap=max(initial_gap,float(abs(np.asarray(x)-np.asarray(y)).max()))
    assert initial_gap<1e-12,initial_gap
    fourier_identity=None;fourier_gap=None;fourier_rounded_match=None
    if architecture_changes and set(architecture_changes)<={'width','heads','input_dims','output_scale','time_features','time_max_frequency'} and c.train.seed==old.train.seed:
        old_template=build_network(old,jax.random.key(0))
        old_initial=eqx.tree_deserialise_leaves(prior/'model_initial.eqx',old_template)
        first=np.asarray(a.frequencies);second=np.asarray(old_initial.frequencies)
        if c.train.compute_dtype!=old.train.compute_dtype:
            np.testing.assert_array_equal(first,second.astype(first.dtype));fourier_rounded_match=True
        else:np.testing.assert_allclose(first,second,rtol=0,atol=1e-12)
        fourier_identity=bool(np.array_equal(first,second))
        fourier_gap=float(abs(first-second).max())
    from nonlocal_pinn.training.optimizers import make_optimizer
    state_template=make_optimizer(c).init(eqx.filter(a,eqx.is_array))
    state=eqx.tree_deserialise_leaves(run/'optimizer_final.eqx',state_template)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state) if eqx.is_array(x))
    expected_dtype=np.dtype(c.train.compute_dtype)
    for name,tree in (('model',a),('optimizer',state),('training_bank',bank)):
        for leaf in jax.tree.leaves(tree):
            if eqx.is_inexact_array(leaf):assert leaf.dtype==expected_dtype,(name,leaf.shape,leaf.dtype)
    imported_state_gap=None
    if start_step:
        parent=Path(c.train.continuation.parent_run)
        imported=eqx.tree_deserialise_leaves(run/'optimizer_initial.eqx',state_template)
        expected_state=eqx.tree_deserialise_leaves(parent/'optimizer_final.eqx',state_template)
        imported_state_gap=0.
        for x,y in zip(jax.tree.leaves(imported),jax.tree.leaves(expected_state),strict=True):
            if eqx.is_array(x):
                np.testing.assert_array_equal(x,y)
    parameter_count=sum(np.size(x) for x in jax.tree.leaves(eqx.filter(a,eqx.is_array)))-np.size(a.frequencies)
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
    predict=eqx.filter_jit(lambda n,d,t:fields(n,d,t,c));maximum=0.;field_gaps=np.zeros(2)
    for i in labeled:
        d=refs[i];item=jax.tree.map(lambda x:x[i],bank)
        pred=np.asarray(predict(model,item,jnp.asarray(d['time'])))[:,:len(d['points'])]
        with np.load(run/f'prediction_{i:03d}.npz',allow_pickle=False) as f:
            assert f['phi'].dtype==f['mu'].dtype==expected_dtype
            gaps=np.max(abs(pred-np.stack((f['phi'],f['mu']),axis=-1)),axis=(0,1))
            field_gaps=np.maximum(field_gaps,gaps);maximum=max(maximum,float(gaps.max()))
    tolerances=np.array([2e-5,2e-4]) if c.train.compute_dtype=='float32' else np.array([1e-9,1e-9])
    assert np.all(field_gaps<tolerances),(field_gaps,tolerances)
    with np.load(run/'training_draws.npz',allow_pickle=False) as f:ids=f['condition_ids']
    with np.load(prior/'training_draws.npz',allow_pickle=False) as f:old_ids=f['condition_ids']
    assert ids.shape==(c.train.iterations-start_step,c.operator.condition_batch)
    key=jax.random.split(jax.random.key(c.train.seed),3)[1]
    expected_ids=jax.vmap(lambda i:jax.random.randint(jax.random.fold_in(key,i),(c.operator.condition_batch,),0,c.operator.train_conditions))(jnp.arange(start_step+1,c.train.iterations+1,dtype=jnp.uint32))
    np.testing.assert_array_equal(ids,np.asarray(expected_ids))
    prefix=min(len(ids),len(old_ids)) if start_step==0 and ids.shape[1]==old_ids.shape[1] and not bank_growth else 0
    if prefix:np.testing.assert_array_equal(ids[:prefix],old_ids[:prefix])
    assert set(np.unique(ids))==set(range(c.operator.train_conditions))
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
    counts=np.bincount(ids.ravel(),minlength=c.operator.train_conditions)
    phase_now=c.phase.to_dict() if 'phase' in c else None
    phase_before=old.phase.to_dict() if 'phase' in old else None
    rhs_now=bool(initial_rhs_enabled(c));rhs_before=bool(initial_rhs_enabled(old))
    width_only=(set(architecture_changes)=={'width'} and rhs_now==rhs_before and not sampling_changes and not training_changes
        and not metric_changes and c.operator.condition_batch==old.operator.condition_batch and phase_now==phase_before and not bank_growth)
    report=dict(run=str(run),prior_run=str(prior),config_matches_explicit_preset=True,
        unchanged_physics_geometry_input_operators=True,bank_growth=growth,
        residual_metric_changes=metric_changes,
        initial_rhs_condition=dict(before=rhs_before,after=rhs_now),
        sampling_changes=json.loads(ConfigDict(sampling_changes).to_json()),
        architecture_changes=architecture_changes,
        training_changes=json.loads(ConfigDict(training_changes).to_json()),
        phase_parameterization=phase_now,
        expected_preset=preset,trainable_parameters=parameter_count,
        comparison_scope='Backbone-width-only control; matched objective/sampler/optimizer and common seed' if width_only else 'Same physical benchmark; bundled architecture/settings optimization, not an isolated feature-count ablation',
        fixed_spatial_fourier_arrays_identical_to_prior=fourier_identity,
        fixed_spatial_fourier_arrays_equal_after_compute_rounding=fourier_rounded_match,
        fixed_spatial_fourier_maximum_difference_to_prior=fourier_gap,
        initial_parameter_policy='parent checkpoint' if start_step else 'common fresh seed',
        initial_parameter_gap=initial_gap,start_step=start_step,optimizer_state_finite=True,
        imported_optimizer_state_gap=imported_state_gap,
        condition_batch=c.operator.condition_batch,prior_condition_batch=old.operator.condition_batch,
        expected_initial_pde_fraction_preserved=True,initial_pde_fraction=initial_fraction,
        condition_draws_match_global_seed=True,actual_common_condition_draws_identical=bool(prefix),common_draw_prefix_steps=prefix,
        no_heldout_geometry_draws=True,visits_min=int(counts.min()),visits_max=int(counts.max()),
        initial_modes=10,initial_analytic_bound=float(c.initial.amplitude*abs(wave[:,2]).sum()),
        shortest_mode_wavelength=float(2*np.pi/np.sqrt(k2.max())),mesh_size=c.geometry.mesh_size,
        local_linearized_modal_growth=(c.physical.mobility*k2*(c.physical.beta-c.physical.epsilon**2*k2)).tolist(),
        optimizer_payload_has_no_future_fields=True,cpu_gpu_maximum_phi_mu_difference=maximum,
        model_optimizer_training_data_dtype=str(expected_dtype),
        cpu_gpu_field_gaps=field_gaps.tolist(),cpu_gpu_field_tolerances=tolerances.tolist(),
        native_time_phi_max=max(r['native_time_errors']['phi'] for r in manifest['conditions'] if r['has_reference']),
        native_time_mu_max=max(r['native_time_errors']['mu'] for r in manifest['conditions'] if r['has_reference']),
        local_reference_checks=local,reference_terminal_phase_statistics=physical,
        native_test=evaluation['test'],local_FEM=evaluation['local_FEM'],
        wall_seconds=summary['wall_seconds'],
        cumulative_wall_seconds=summary['cumulative_wall_seconds'] if 'cumulative_wall_seconds' in summary else summary['wall_seconds'],
        new_updates=c.train.iterations-start_step,warm_update_seconds=summary['mean_update_seconds_after_100'],
        memory=summary['device_memory_stats'],scope=f'One cold seed, finite{c.operator.train_conditions}/8development cohort; native-only, local FEM deferred')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--bank-growth',action='store_true',help='Explicitly verify expanded input bank and exact original references')
    a=p.parse_args();audit(a.run,a.prior,a.output,a.config,a.bank_growth)
