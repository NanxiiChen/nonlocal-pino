"""Deterministic full-time L-BFGS, with explicitly inherited SOAP provenance."""
from pathlib import Path
import json
import shutil
import time
from datetime import datetime,timezone
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from ml_collections import ConfigDict
from scipy.optimize import minimize
from examples.channel_nsch2d.losses import make_loss_fn,part_names
from examples.channel_nsch2d.network import build_network,encode,fields,extend_time_basis,extend_stress_outputs
from examples.channel_nsch2d.problem import identity
from examples.channel_nsch2d.stress_initialization import initializer_paths,import_initializer,FILENAMES
from nonlocal_pinn.training.runtime import configure_gpu_policy


def make_accumulated_objective(model,loss,data,nodes,time_batch):
    """Each call differentiates the same current parameters at EVERY time node."""
    nodes=np.asarray(nodes)
    if isinstance(time_batch,bool) or not isinstance(time_batch,int) or time_batch<1:
        raise ValueError('Positive integer time_batch required')
    if (nodes.ndim!=1 or len(nodes)<time_batch or len(nodes)%time_batch
            or not np.isfinite(nodes).all() or np.any(np.diff(nodes)<=0)):
        raise ValueError('Ordered finite time nodes must form complete fixed-size batches')
    parameters,static=eqx.partition(model,eqx.is_inexact_array)
    initial,unravel=ravel_pytree(parameters)
    restore=lambda vector:eqx.combine(unravel(jnp.asarray(vector,dtype=initial.dtype)),static)
    def objective(vector,batch,times):return loss(restore(vector),batch,times)
    value_grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
    blocks=jnp.asarray(nodes.reshape(-1,time_batch))
    def evaluate(vector):
        vector=jnp.asarray(vector,dtype=initial.dtype)
        value=0.;gradient=np.zeros(initial.shape,dtype=np.float64);parts=None
        for t in blocks:
            (v,p),g=value_grad(vector,data,t)
            v=float(v);p=np.asarray(p,dtype=np.float64);g=np.asarray(g,dtype=np.float64)
            if not np.isfinite(v) or not np.isfinite(p).all() or not np.isfinite(g).all():
                raise FloatingPointError('Nonfinite full-time L-BFGS objective/gradient')
            value+=v;gradient+=g;parts=p if parts is None else parts+p
        return value/len(blocks),gradient/len(blocks),parts/len(blocks)
    return np.asarray(initial,dtype=np.float64),restore,evaluate


def time_expansion(c,saved):
    current=json.loads(c.to_json())
    if current['arch']['time_features']==saved['arch']['time_features']:return False
    if 'time_embedding' in saved or 'time_embedding' not in current:
        raise ValueError('Time expansion requires an original linear-time parent and explicit new basis')
    spec=current['time_embedding']
    if (set(spec)!={'kind','scale','initialization'} or spec['kind']!='linear_log' or spec['initialization']!='preserve_linear'
            or not np.isfinite(spec['scale']) or spec['scale']<=0
            or current['arch']['time_features']!=2*saved['arch']['time_features']):
        raise ValueError('Only a declared function-preserving linear-to-extended-log time expansion is supported')
    return True


def stress_expansion(c,saved):
    current=json.loads(c.to_json())
    if ('stress_formulation' in current)==('stress_formulation' in saved):return False
    if 'stress_formulation' in saved:raise ValueError('Cannot remove the parent stress representation')
    s=current['stress_formulation']
    if (set(s)!={'kind','initialization','constitutive_scale'} or s['kind']!='cauchy_mixed'
            or s['initialization']!='preserve_original' or not np.isfinite(s['constitutive_scale'])
            or s['constitutive_scale']<=0 or 'velocity' not in saved or 'pressure_coordinate' in saved
            or time_expansion(c,saved)):
        raise ValueError('Only declared physical-function-preserving Cauchy stress extension is supported')
    return True


def parent_spec(c):
    parent=Path(c.train.refinement.parent_run)
    if not parent.is_absolute():raise ValueError('Refinement parent_run must be absolute')
    for name in ('config.json','summary.json','validation.json','known_inputs.npz',
                 'sampling.npz','model_final.eqx','optimizer_final.eqx','prediction.npz','operators.json'):
        if not (parent/name).is_file():raise FileNotFoundError(parent/name)
    saved=json.loads((parent/'config.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if summary['status']!='completed' or json.loads((parent/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed':
        raise ValueError('Refinement requires a complete independently replayed SOAP parent')
    if saved['train']['iterations']!=c.train.iterations or summary['iterations']!=c.train.iterations:
        raise ValueError('Refinement SOAP update count must equal the completed parent')
    current=json.loads(c.to_json());original=json.loads(json.dumps(saved))
    if time_expansion(c,saved):
        del current['time_embedding']
        current['arch']['time_features']=original['arch']['time_features']
    stress_added=stress_expansion(c,saved)
    if 'droplet_metric' in c:
        if 'droplet_metric' in original or time_expansion(c,saved) or stress_added:
            raise ValueError('First frozen droplet control only adds a metric to an unchanged original parent')
        del current['droplet_metric']
    if 'stress_initializer' in c.train.refinement and not stress_added:
        raise ValueError('Auxiliary initializer requires an original four-output parent')
    if stress_added:del current['stress_formulation']
    for config in (current,original):
        del config['log']
        for key in ('continuation','refinement'):
            if key in config['train']:del config['train'][key]
    if original!=current:raise ValueError('Refinement must preserve every parent numerical/physical setting')
    return parent,summary


def audit_refinement(run,c):
    """Input records are inherited; the only new support is the full-time grid."""
    run=Path(run);parent,summary=parent_spec(c)
    saved=json.loads((parent/'config.json').read_text())
    time_added=time_expansion(c,saved);stress_added=stress_expansion(c,saved);expanded=time_added or stress_added
    for child,old in ((('model_parent.eqx' if expanded else 'model_initial.eqx'),'model_final.eqx'),('optimizer_soap.eqx','optimizer_final.eqx'),
                      ('known_inputs.npz','known_inputs.npz'),('sampling.npz','sampling.npz')):
        if (run/child).read_bytes()!=(parent/old).read_bytes():
            raise ValueError(f'Inherited refinement payload differs: {child}')
    if expanded:
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
        old=eqx.tree_deserialise_leaves(parent/'model_final.eqx',build_network(ConfigDict(saved),key))
        transfer=extend_stress_outputs if stress_added else extend_time_basis
        expected=transfer(old,build_network(c,key))
        if 'stress_initializer' in c.train.refinement:
            original=initializer_paths(c.train.refinement.stress_initializer)
            local={key:run/name for key,name in FILENAMES.items()}
            for name in FILENAMES:
                if local[name].read_bytes()!=original[name].read_bytes():
                    raise ValueError(f'Copied stress initializer differs: {name}')
            expected=import_initializer(old,build_network(c,key),c,local)
        actual=eqx.tree_deserialise_leaves(run/'model_initial.eqx',build_network(c,key))
        for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual),strict=True):
            if eqx.is_array(a):np.testing.assert_array_equal(a,b)
    with np.load(run/'refinement_nodes.npz',allow_pickle=False) as f:nodes=f['time']
    count=int(round(c.physical.terminal_time/c.time_discretization.dt))
    expected=np.arange(1,count+1,dtype=c.train.compute_dtype)*np.asarray(c.time_discretization.dt,dtype=c.train.compute_dtype)
    np.testing.assert_array_equal(nodes,expected)
    final=json.loads((run/'refinement_summary.json').read_text())
    rows=[json.loads(s) for s in (run/'refinement_evaluations.jsonl').read_text().splitlines()]
    if (not rows or len(rows)!=final['objective_evaluations']
            or [r['evaluation'] for r in rows]!=list(range(1,len(rows)+1))
            or any(r['time_nodes']!=count for r in rows)):
        raise ValueError('Incomplete actual full-time objective support records')
    if json.loads((run/'summary.json').read_text())['refinement']['iterations']!=final['iterations']:
        raise ValueError('Refinement accepted-step count disagrees')
    metric_audit=None
    if 'droplet_metric' in c:
        from examples.channel_nsch2d.droplet_metric import parent_phase,weights_from_phase,metric_summary
        with np.load(run/'frozen_droplet_metric.npz',allow_pickle=False) as f:
            if set(f.files)!={'phase','weights','time'}:raise ValueError('Unexpected frozen droplet metric payload')
            phase,weight,metric_time=f['phase'],f['weights'],f['time']
        np.testing.assert_array_equal(metric_time,nodes)
        with np.load(parent/'known_inputs.npz',allow_pickle=False) as f:
            raw={k:jnp.asarray(f[k][0]) for k in f.files}
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
        old=eqx.tree_deserialise_leaves(parent/'model_final.eqx',build_network(ConfigDict(saved),key))
        actual=parent_phase(old,raw,nodes,c)
        gap=float(np.max(np.abs(actual-phase)))
        if gap>2e-5:raise ValueError(f'Frozen parent phase reconstruction failed:{gap}')
        measure=np.asarray(raw['transport_measure']*raw['interior'])
        expected=weights_from_phase(phase,measure,c.droplet_metric).astype(c.train.compute_dtype)
        np.testing.assert_array_equal(weight,expected)
        metric_audit=metric_summary(phase,weight,measure)
        metric_audit.update(parent_phase_max_gap=gap,full_BDF_nodes_reconstructed=True,reference_fields_used=False)
    return dict(status='inherited_parent_and_full_time_objective_support_passed',parent_run=str(parent),
        parent_soap_iterations=summary['iterations'],new_soap_updates=0,
        time_nodes=count,time_batch=c.train.refinement.time_batch,objective_evaluations=len(rows),
        inherited_sampling_scope='Parent stochastic stage only; no new stochastic draws',
        function_preserving_time_expansion=time_added,function_preserving_stress_expansion=stress_added,
        auxiliary_stress_initializer_reconstructed='stress_initializer' in c.train.refinement,
        frozen_droplet_metric=metric_audit,
        optimizer_resume_scope='Parent SOAP state retained separately; exact SciPy L-BFGS restart is not supported')


def train_refinement(c):
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    parent,parent_summary=parent_spec(c);s=c.train.refinement
    saved=json.loads((parent/'config.json').read_text())
    time_added=time_expansion(c,saved);stress_added=stress_expansion(c,saved);expanded=time_added or stress_added
    gate=json.loads((Path(c.reference.root)/'reference-validation.json').read_text())
    if gate['status']!='passed' or gate['identity']!=json.loads(json.dumps(identity(c))):
        raise ValueError('Refinement requires the unchanged independently validated physical reference')
    with np.load(parent/'known_inputs.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    with np.load(parent/'prediction.npz',allow_pickle=False) as f:reference={k:f[k] for k in f.files}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(parent/'model_final.eqx',build_network(ConfigDict(saved),key))
    if expanded:
        transfer=extend_stress_outputs if stress_added else extend_time_basis
        if 'stress_initializer' in s:
            model=import_initializer(model,build_network(c,key),c,initializer_paths(s.stress_initializer))
        else:model=transfer(model,build_network(c,key))
    single=jax.tree.map(lambda a:a[0],data)
    points=jnp.asarray(reference['points'],dtype=c.train.compute_dtype)
    times=jnp.asarray(reference['time'],dtype=c.train.compute_dtype)
    predict=eqx.filter_jit(lambda n:fields(n,encode(n,single),points,times,single['amplitudes'],c))
    initial_fields=np.asarray(predict(model),dtype=np.float64)
    replay=float(np.max(np.abs(initial_fields-reference['fields'])))
    if not np.isfinite(initial_fields).all() or replay>2e-5:raise ValueError(f'Refinement parent replay failed: {replay}')
    name=c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-channel-fulltime-lbfgs'
    run=Path(c.log.log_dir)/name;run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(c.to_json(indent=2)+'\n')
    for old,new in (('known_inputs.npz','known_inputs.npz'),('sampling.npz','sampling.npz'),
                    ('model_final.eqx','model_parent.eqx' if expanded else 'model_initial.eqx'),('optimizer_final.eqx','optimizer_soap.eqx'),
                    ('operators.json','operators.json')):shutil.copyfile(parent/old,run/new)
    if expanded:eqx.tree_serialise_leaves(run/'model_initial.eqx',model)
    if 'stress_initializer' in s:
        for name,path in initializer_paths(s.stress_initializer).items():shutil.copyfile(path,run/FILENAMES[name])
    shutil.copytree(Path(__file__).parent,run/'source/examples/channel_nsch2d',ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/training/runtime.py'):
        destination=run/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[2]/relative,destination)
    provenance=dict(parent_run=str(parent),parent_replay_max_gap=replay,
        parent_soap_updates=parent_summary['iterations'],new_stochastic_updates=0,
        optimizer='L-BFGS-B',settings=s.to_dict(),input_scope='Exact parent input/sampling bytes; full positive BDF node grid',
        exact_optimizer_resume_supported=False)
    provenance['function_preserving_time_expansion']=time_added
    provenance['function_preserving_stress_expansion']=stress_added
    if 'droplet_metric' in c:
        provenance['residual_metric']=dict(settings=c.droplet_metric.to_dict(),
            source='Exact parent predictions at all positive BDF nodes; fixed during optimization',
            payload='frozen_droplet_metric.npz',reference_fields_used=False)
    if time_added:
        old=saved['arch']['time_features']
        provenance['time_basis_transfer']=dict(linear_features_retained=old,log_features_added=old,
            time_mlp_input_before=1+2*old,time_mlp_input_after=1+4*old,
            added_input_weights=c.arch.width*2*old,new_log_columns_initialized_zero=True,
            all_other_parameter_arrays_copied=True)
    if stress_added:
        provenance['stress_output_transfer']=dict(old_channels=4,new_channels=6,
            pressure='minus_half_trace',initial_stress='minus_parent_pressure_times_identity',
            added_parameters=2*(c.arch.width+1),all_nonoutput_arrays_copied=True,
            optimizer_state_scope='Parent SOAP provenance only; L-BFGS starts fresh history')
        if 'stress_initializer' in s:
            provenance['stress_output_transfer']['initial_stress']='verified_physics_auxiliary_block'
            provenance['stress_output_transfer']['initializer']=s.stress_initializer.to_dict()
    (run/'refinement.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    count=int(round(c.physical.terminal_time/c.time_discretization.dt))
    nodes=np.arange(1,count+1,dtype=c.train.compute_dtype)*np.asarray(c.time_discretization.dt,dtype=c.train.compute_dtype)
    np.savez_compressed(run/'refinement_nodes.npz',time=nodes)
    if 'droplet_metric' in c:
        from examples.channel_nsch2d.droplet_metric import parent_phase,weights_from_phase,metric_summary
        print(json.dumps(dict(stage='freezing_parent_droplet_metric',time_nodes=count)),flush=True)
        preparation_start=time.perf_counter()
        phase=parent_phase(model,single,nodes,c)
        measure=np.asarray(single['transport_measure']*single['interior'])
        weights=weights_from_phase(phase,measure,c.droplet_metric).astype(c.train.compute_dtype)
        np.savez_compressed(run/'frozen_droplet_metric.npz',phase=phase,weights=weights,time=nodes)
        (run/'frozen_droplet_metric.json').write_text(json.dumps(dict(settings=c.droplet_metric.to_dict(),
            preparation_seconds=time.perf_counter()-preparation_start,
            **metric_summary(phase,weights,measure)),indent=2,allow_nan=False)+'\n')
        data=dict(data,droplet_weights=jnp.asarray(weights[None]))
    initial,restore,evaluate=make_accumulated_objective(model,make_loss_fn(c),data,nodes,s.time_batch)
    start=time.perf_counter();calls=0;accepted_count=0;latest=None;final_fields=initial_fields
    def assess(candidate,iteration):
        nonlocal final_fields
        final_fields=np.asarray(predict(candidate),dtype=np.float64)
        if not np.isfinite(final_fields).all():raise FloatingPointError('Nonfinite refined physical fields')
        target=reference['reference'];relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        metrics=dict(step=c.train.iterations+iteration,refinement_step=iteration,elapsed=time.perf_counter()-start,
            phi=relative(final_fields[...,0],target[...,0]),terminal_phi=relative(final_fields[-1,:,0],target[-1,:,0]),
            dispersed_phase=relative((1-final_fields[...,0])/2,(1-target[...,0])/2),
            velocity=relative(final_fields[1:,:,2:4],target[1:,:,2:4]),pressure=relative(final_fields[1:,:,4],target[1:,:,4]),
            chemical=relative(final_fields[1:,:,1],target[1:,:,1]),reference=gate['reference'],
            scope='One development FEM geometry; no evolved labels in the full-time objective')
        with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        print(json.dumps(metrics),flush=True);return metrics
    metrics=assess(model,0)
    with (run/'refinement_evaluations.jsonl').open('x') as objective_log,(run/'loss.jsonl').open('x') as accepted_log:
        def objective(vector):
            nonlocal calls,latest
            if latest is not None and np.array_equal(vector,latest[0]):return latest[1],latest[2]
            v,g,p=evaluate(vector);calls+=1
            latest=(vector.copy(),v,g,p)
            row=dict(evaluation=calls,total=v,gradient_inf=float(np.max(np.abs(g))),time_nodes=count,
                elapsed=time.perf_counter()-start,parts=dict(zip(part_names(c),p.tolist(),strict=True)))
            objective_log.write(json.dumps(row,allow_nan=False)+'\n');objective_log.flush()
            if calls==1:print(json.dumps(dict(stage='first_full_objective_completed',**row)),flush=True)
            return v,g
        def accepted(vector):
            nonlocal accepted_count,metrics
            objective(vector);accepted_count+=1
            row=dict(step=c.train.iterations+accepted_count,refinement_step=accepted_count,total=latest[1],
                parts=dict(zip(part_names(c),latest[3].tolist(),strict=True)),elapsed=time.perf_counter()-start)
            accepted_log.write(json.dumps(row,allow_nan=False)+'\n');accepted_log.flush()
            if accepted_count==1 or accepted_count%10==0:print(json.dumps(row),flush=True)
            if accepted_count%c.log.eval_every==0:metrics=assess(restore(vector),accepted_count)
            if accepted_count%c.log.checkpoint_every==0:eqx.tree_serialise_leaves(run/f'ckpt_{c.train.iterations+accepted_count:06d}.eqx',restore(vector))
        print(json.dumps(dict(stage='compiling_full_time_gradient',time_nodes=count,time_batch=s.time_batch)),flush=True)
        result=minimize(objective,initial,method='L-BFGS-B',jac=True,callback=accepted,
            options=dict(maxiter=s.iterations,maxfun=s.max_evaluations,maxls=s.max_line_search,maxcor=s.history_size,
                         gtol=s.gradient_tolerance,ftol=s.function_tolerance))
    if (accepted_count!=result.nit or not np.isfinite(result.x).all()
            or not np.isfinite(result.jac).all() or not np.isfinite(result.fun)):
        raise ValueError('L-BFGS final iterate/count is invalid')
    model=restore(result.x)
    if metrics['refinement_step']!=result.nit:metrics=assess(model,int(result.nit))
    eqx.tree_serialise_leaves(run/'model_final.eqx',model)
    np.savez_compressed(run/'optimizer_lbfgs_final.npz',position=result.x,gradient=result.jac)
    np.savez_compressed(run/'prediction.npz',fields=final_fields,points=reference['points'],time=reference['time'],
        amplitudes=reference['amplitudes'],reference=reference['reference'])
    refinement=dict(optimizer='L-BFGS-B',iterations=int(result.nit),success=bool(result.success),status=int(result.status),
        message=str(result.message),objective_evaluations=calls,scipy_function_evaluations=int(result.nfev),
        final_loss=float(result.fun),gradient_inf=float(np.max(np.abs(result.jac))),time_nodes=count,time_batch=s.time_batch,
        neural_dtype=c.train.compute_dtype,host_accumulation_dtype='float64',exact_resume_supported=False)
    (run/'refinement_summary.json').write_text(json.dumps(refinement,indent=2,allow_nan=False)+'\n')
    wall=time.perf_counter()-start
    lifetime=parent_summary['cumulative_wall_seconds'] if 'cumulative_wall_seconds' in parent_summary else parent_summary['wall_seconds']
    summary=dict(status='completed',stage_kind='deterministic_full_time_refinement',iterations=c.train.iterations+int(result.nit),
        soap_updates=c.train.iterations,
        start_step=parent_summary['start_step'] if 'start_step' in parent_summary else 0,new_stochastic_updates=0,
        parent_run=str(parent),wall_seconds=wall,cumulative_wall_seconds=lifetime+wall,metrics=metrics,
        train_conditions=1,supervised_evolved_field_samples=0,refinement=refinement,
        inherited_sampling_scope='Parent stochastic records; no new random draws',
        device_memory_bytes=jax.devices()[0].memory_stats(),
        device=dict(kind=jax.devices()[0].device_kind,platform=jax.default_backend(),jax_version=jax.__version__))
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True);return run
