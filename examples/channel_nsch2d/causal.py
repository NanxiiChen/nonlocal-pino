"""Coupled causal weights span the full sampled time batch, not microbatches."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.training.causal import CausalWeightor
from examples.channel_nsch2d.losses import make_time_parts_fn,part_names


def epsilon_at(step,c):
    """Host schedule; the selected value enters JIT as a dynamic scalar."""
    if isinstance(step,bool) or not isinstance(step,int) or not 0<=step<c.train.iterations:
        raise ValueError('Causal schedule requires an actual zero-based optimizer step')
    return c.causal.epsilon_values[step//c.causal.steps_per_epsilon]


def make_epsilon_update(c):
    """Reuse shared feedback per geometry; all must pass before increasing."""
    settings=c.causal.adaptation
    controller=CausalWeightor(n_chunks=None)

    def update(weights,epsilon):
        if weights.shape!=(c.operator.condition_batch,c.sample.Nt) or epsilon.shape!=():
            raise ValueError('Causal feedback requires condition/time weights and scalar epsilon')
        proposed=jax.vmap(lambda w:controller.update_eps(w,epsilon,
            settings.step_size,settings.max_eps,settings.max_last_weight,
            settings.min_mean_weight))(weights)
        return jax.lax.stop_gradient(jnp.min(proposed))

    return update


def _audit_feedback(run,c):
    """NumPy replay of EVERY used/next epsilon, independent of JAX controller."""
    rows=[json.loads(s) for s in (Path(run)/'causal-controller.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(1,c.train.iterations+1)):
        raise ValueError('Incomplete actual causal feedback history')
    cfg=c.causal.adaptation;dtype=np.dtype(c.train.compute_dtype)
    cast=dtype.type;epsilon=cast(cfg.eps_init);maximum_gap=0.;counts=dict(increase=0,decrease=0,hold=0)
    for row in rows:
        mean=np.asarray(row['mean_weights'],dtype=dtype);last=np.asarray(row['last_time_weights'],dtype=dtype)
        shape=(c.operator.condition_batch,)
        if (mean.shape!=shape or last.shape!=shape or not np.isfinite(mean).all()
                or not np.isfinite(last).all() or np.any(last<0) or np.any(last>mean+2e-6)
                or np.any(mean<1/c.sample.Nt-2e-6) or np.any(mean>1)):
            raise ValueError('Invalid pergeometry causal feedback statistics')
        used=cast(row['epsilon']);next_value=cast(row['next_epsilon'])
        if not np.isfinite(used) or not np.isfinite(next_value) or min(used,next_value)<=0:
            raise ValueError('Causal feedback epsilon must stay finite and positive')
        proposed=np.where((last>cast(cfg.max_last_weight))&(used<cast(cfg.max_eps)),
            used*cast(cfg.step_size),used)
        proposed=np.where(mean<cast(cfg.min_mean_weight),proposed/cast(cfg.step_size),proposed)
        expected=np.clip(proposed,cast(0),cast(cfg.max_eps)).min()
        gap=max(abs(float(used)-float(epsilon)),abs(float(next_value)-float(expected)))
        maximum_gap=max(maximum_gap,gap)
        if not np.isclose(used,epsilon,rtol=2e-6,atol=1e-10) or not np.isclose(next_value,expected,rtol=2e-6,atol=1e-10):
            raise ValueError(f'Causal feedback state transition disagrees at update{row["step"]}')
        action='increase' if next_value>used else ('decrease' if next_value<used else 'hold')
        counts[action]+=1;epsilon=next_value
    summary=json.loads((Path(run)/'summary.json').read_text())
    np.testing.assert_allclose(summary['causal']['final_epsilon'],rows[-1]['epsilon'],rtol=2e-6,atol=1e-10)
    np.testing.assert_allclose(summary['causal']['next_epsilon'],rows[-1]['next_epsilon'],rtol=2e-6,atol=1e-10)
    return rows,dict(status='every_actual_feedback_state_transition_passed',updates=len(rows),
        maximum_state_gap=maximum_gap,actions=counts,final_next_epsilon=float(epsilon),
        reduction='Minimum of independent geometry proposals; no averaging hides an underweighted sample')


def make_causal_fn(c):
    per_time=make_time_parts_fn(c)
    controller=CausalWeightor(n_chunks=None)
    size=c.train.microbatch.size;blocks=c.sample.Nt//size

    def block_loss(model,batch,ts,w):
        parts=jnp.mean(per_time(model,batch,ts)*w[:,:,None],axis=(0,1))
        return jnp.sum(parts),parts

    value_grad=eqx.filter_value_and_grad(block_loss,has_aux=True)

    def objective(model,batch,times,epsilon):
        if times.shape!=(c.sample.Nt,):raise ValueError('Causal requires the complete configured time batch')
        times=eqx.error_if(times,jnp.any(~jnp.isfinite(times))|jnp.any(jnp.diff(times)<=0),
            'Causal times must be finite and strictly increasing')
        ts=times.reshape(blocks,size)
        raw=jax.lax.map(lambda t:per_time(model,batch,t),ts)
        raw=jnp.transpose(raw,(1,0,2,3)).reshape(raw.shape[1],len(times),raw.shape[-1])
        segments=jnp.sum(raw,axis=-1)
        weights=jax.vmap(lambda s:controller.compute_weights(s,epsilon))(segments)
        wblocks=jnp.transpose(weights.reshape(len(weights),blocks,size),(1,0,2))
        zero=jax.tree.map(jnp.zeros_like,eqx.filter(model,eqx.is_inexact_array))
        initial=(jnp.zeros((),dtype=times.dtype),jnp.zeros(len(part_names(c)),dtype=times.dtype),zero)

        def add(carry,payload):
            total,parts,gradient=carry;t,w=payload
            (value,vector),grad=value_grad(model,batch,t,w)
            gradient=jax.tree.map(lambda a,b:a+b,gradient,grad)
            return (total+value,parts+vector,gradient),None

        (total,parts,gradient),_=jax.lax.scan(add,initial,(ts,wblocks))
        gradient=jax.tree.map(lambda g:g/blocks,gradient)
        aux=dict(weights=weights,segments=segments,unweighted_parts=jnp.mean(raw,axis=(0,1)))
        return total/blocks,parts/blocks,gradient,aux

    return objective


def audit_trace(run,c):
    """Independent NumPy prefix/schedule/objective reconstruction from logs."""
    rows=[json.loads(s) for s in (Path(run)/'causal.jsonl').read_text().splitlines()]
    adaptive='adaptation' in c.causal
    feedback,feedback_audit=_audit_feedback(run,c) if adaptive else (None,None)
    expected=sorted({1,c.train.iterations,*range(100,c.train.iterations+1,100)})
    if not adaptive:expected=sorted({*expected,*range(1,c.train.iterations+1,c.causal.steps_per_epsilon)})
    if [r['step'] for r in rows]!=expected:raise ValueError('Incomplete actual causal diagnostics')
    max_gap=0.
    for row in rows:
        eps=feedback[row['step']-1]['epsilon'] if adaptive else epsilon_at(row['step']-1,c)
        if not np.isclose(row['epsilon'],eps,rtol=1e-6,atol=0):raise ValueError('Causal epsilon schedule disagrees')
        w=np.asarray(row['weights']);seg=np.asarray(row['segments'])
        shape=(c.operator.condition_batch,c.sample.Nt)
        if w.shape!=shape or seg.shape!=shape or not np.isfinite(w).all() or not np.isfinite(seg).all():
            raise ValueError('Invalid causal condition/time diagnostics')
        if np.any(seg<0) or np.any(w<0) or np.any(w>1) or np.any(np.diff(w,axis=-1)>1e-7):
            raise ValueError('Causal segments/weights violate positive-loss temporal prefix assumptions')
        expected_w=np.exp(-row['epsilon']*np.c_[np.zeros(len(seg)),np.cumsum(seg[:,:-1],axis=-1)])
        gap=float(np.max(np.abs(w-expected_w)));max_gap=max(max_gap,gap)
        if gap>2e-5 or not np.array_equal(w[:,0],np.ones(len(w))):raise ValueError('Global stopped prefix weights disagree')
        if adaptive:
            state=feedback[row['step']-1]
            np.testing.assert_allclose(w.mean(axis=-1),state['mean_weights'],rtol=2e-6,atol=2e-7)
            np.testing.assert_array_equal(w[:,-1],state['last_time_weights'])
            np.testing.assert_allclose(row['next_epsilon'],state['next_epsilon'],rtol=2e-6,atol=1e-10)
        np.testing.assert_allclose(np.mean(w*seg),row['total'],rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(np.sum(row['unweighted_parts']),np.mean(seg),rtol=2e-5,atol=2e-6)
    result=dict(status='actual_causal_schedule_global_prefix_and_objective_passed',records=len(rows),
        maximum_weight_gap=max_gap,first_weight_one=True,geometry_prefixes_separate=True,
        prefix_scope='All sampled times before microbatch slicing',
        scope='Logged pre-update objectives; stopped AD verified by focused gradient tests, not inferred from logs')
    if adaptive:result['feedback']=feedback_audit
    return result


def comparison_config(c,control):
    """Audit one explicit causal, temporal, or stream-envelope change."""
    control=Path(control)
    if not control.is_absolute():raise ValueError('Causal comparison control must be an absolute path')
    for name in ('config.json','summary.json','validation.json','sampling.npz','prediction.npz',
                 'known_inputs.npz','known_inputs_final.npz','model_initial.eqx','optimizer_initial.eqx'):
        if not (control/name).is_file():raise FileNotFoundError(control/name)
    saved=json.loads((control/'config.json').read_text())
    summary=json.loads((control/'summary.json').read_text())
    if (summary['status']!='completed' or summary['iterations']!=c.train.iterations
            or json.loads((control/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed'):
        raise ValueError('Causal requires a matching completed/audited update-budget control')
    current=json.loads(c.to_json())
    if 'initialization'in c.train:
        from examples.channel_nsch2d.initialization import model_initialization_spec
        spec=model_initialization_spec(c)
        if Path(spec['source_run'])!=control:raise ValueError('Model-only comparison must use its actual physics source run')
        return saved
    if 'chemical_residual_scale'in current['loss']:
        if 'chemical_residual_scale'in saved['loss']:
            raise ValueError('Independent chemical normalization requires the original shared-scale control')
        del current['loss']['chemical_residual_scale']
        current['log']['run_name']=saved['log']['run_name']
        if current!=saved:
            raise ValueError('Chemical comparison may change only chemical_residual_scale and run name')
        return saved
    if 'stream_lifting' in current:
        if ('stream_lifting' in saved or 'time_derivative' not in saved
                or 'causal' not in saved or 'adaptation' not in saved['causal']):
            raise ValueError('Stream lifting comparison requires the completed original-envelope adaptiveJVP control')
        del current['stream_lifting']
    elif 'time_derivative' in current:
        if ('time_discretization' not in saved or saved['time_discretization']['kind']!='bdf2'):
            raise ValueError('TimeJVP comparison requires a matching completed BDF2 control')
        if 'causal'in current and ('causal'not in saved or 'adaptation'not in saved['causal']):
            raise ValueError('TimeJVP comparison requires the matching completed adaptiveBDF control')
        if 'causal'not in current and 'causal'in saved:
            raise ValueError('Uniform timeJVP comparison requires the matching completed uniformBDF control')
        del current['time_derivative'];current['time_discretization']=saved['time_discretization']
    else:del current['causal']
    current['log']['run_name']=saved['log']['run_name']
    if current!=saved:raise ValueError('Comparison may change only one explicit causal, timeJVP or stream lifting setting and run name')
    return saved


def audit_comparison(run,control):
    """Complete actual cold states/input streams and independent saved-array L2."""
    from ml_collections import ConfigDict
    run,control=Path(run),Path(control)
    c=ConfigDict(json.loads((run/'config.json').read_text()));control_config=comparison_config(c,control)
    temporal='time_derivative' in c and 'time_derivative' not in control_config
    stream='stream_lifting' in c
    warm='initialization'in c.train
    chemical='chemical_residual_scale'in c.loss
    if warm:
        from examples.channel_nsch2d.initialization import audit_model_initialization
        initialization=audit_model_initialization(run,c)
    else:
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (run/name).read_bytes()!=(control/name).read_bytes():raise ValueError(f'Causal cold state differs: {name}')
    for name in ('known_inputs.npz','known_inputs_final.npz','sampling.npz'):
        with np.load(run/name,allow_pickle=False) as a,np.load(control/name,allow_pickle=False) as b:
            expected=set(b.files)-{'intervals'} if temporal and name=='sampling.npz' else set(b.files)
            if set(a.files)!=expected:raise ValueError(f'Causal input schema differs: {name}')
            for key in a.files:
                if temporal and name=='sampling.npz' and key=='times':
                    if a[key].shape!=b[key].shape or np.array_equal(a[key],b[key]):
                        raise ValueError('TimeJVP requires matching-size fresh continuous draws, not reused BDF times')
                else:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{name}:{key}')
    temporal_audit=None
    if temporal:
        validated=json.loads((run/'validation.json').read_text())
        if (validated['status']!='replay_and_hard_constraints_passed'
                or not validated['actual_draws_reproduced_from_global_keys']
                or not validated['actual_time_strata_passed']):
            raise ValueError('Actual continuous-time draws require canonical global-key/stratum validation')
        temporal_audit=dict(training='continuous_stratified_timeJVP',control='uniform_nodes_BDF2',
            times_equal=False,actual_continuous_global_key_replay_passed=True,
            conditions_initial_final_geometry_banks_and_all_actual_draws_exact=True)
    with np.load(run/'prediction.npz',allow_pickle=False) as a,np.load(control/'prediction.npz',allow_pickle=False) as b:
        for key in ('reference','points','time','amplitudes'):np.testing.assert_array_equal(a[key],b[key])
    scores=[];maximum_gap=0.
    labels=(('shared_chemical_scale','independent_chemical_residual_scale')if chemical else
        ('physics_JVP_parent','projected_mu_model_refinement')if warm else
        ('jvp_original_stream','jvp_normalized_stream') if stream else
        (('bdf2','jvp') if temporal else ('uniform','causal')))
    for label,path in zip(labels,(control,run),strict=True):
        with np.load(path/'prediction.npz',allow_pickle=False) as f:
            value=f['fields'];target=f['reference']
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        metrics=dict(phi=relative(value[...,0],target[...,0]),
            terminal_phi=relative(value[-1,:,0],target[-1,:,0]),
            velocity=relative(value[1:,:,2:4],target[1:,:,2:4]),
            pressure=relative(value[1:,:,4],target[1:,:,4]))
        saved=json.loads((path/'summary.json').read_text())
        gap=max(abs(metrics[k]-saved['metrics'][k]) for k in metrics);maximum_gap=max(maximum_gap,gap)
        if gap>2e-6:raise ValueError('Causal comparison physical metrics do not reproduce')
        scores.append(dict(label=label,run=str(path),metrics=metrics,
            wall_seconds=saved['wall_seconds'],warm_update_seconds=saved['warm_update_seconds']))
    result=dict(status='full_actual_cold_state_condition_time_and_input_parity_passed',
        control=str(control),run=str(run),updates=c.train.iterations,
        condition_draws=c.train.iterations*c.operator.condition_batch,
        time_draws=c.train.iterations*c.sample.Nt,initial_model_and_complete_optimizer_exact=True,
        initial_and_final_input_banks_exact=True,all_actual_sampling_arrays_exact=not temporal,
        metrics_recomputed_maximum_gap=maximum_gap,results=scores,
        scope='One initialization/reused nominalFEM geometry; no multi-seed or population conclusion')
    if temporal:
        result['status']='matched_cold_state_geometry_stream_and_independent_continuous_time_replay_passed'
        result['temporal_sampling']=temporal_audit
    if stream:
        result['representation']=dict(training=c.stream_lifting.to_dict(),
            control='quadratic_x_over_length',same_actual_continuous_times=True,
            raw_cold_model_and_complete_optimizer_exact=True,
            initial_positive_time_physical_velocity_equal=False,
            same_physical_PDE_BC_and_reference=True)
    if warm:
        result['status']='physics_model_initialization_fresh_optimizer_and_actual_input_streams_passed'
        del result['initial_model_and_complete_optimizer_exact']
        result['model_initialization']=initialization
        result['comparison_scope']='Source5000 plus5000 new global-time physics updates; not an equally trained single-factor control'
    if chemical:
        result['status']='matched_cold_state_all_inputs_and_chemical_row_scaling_only_passed'
        result['chemical_scaling']=dict(output_amplitude_unchanged=c.loss.chemical_scale,
            control_residual_normalizer=control_config['loss']['chemical_scale'],
            trial_residual_normalizer=c.loss.chemical_residual_scale,
            chemical_MSE_multiplier=(control_config['loss']['chemical_scale']/c.loss.chemical_residual_scale)**2,
            boundary_mu_normalizer_unchanged=c.loss.boundary_mu_scale,
            actual_times_and_geometry_exact=True,physical_output_semantics_unchanged=True)
    return result
