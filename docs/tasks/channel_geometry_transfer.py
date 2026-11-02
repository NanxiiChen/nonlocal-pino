"""Frozen geometry transfer with unchanged physical field semantics and FEM gates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def check_transfer(source,target):
    """Allow only wall amplitudes/prior and direct artifact metadata to change."""
    from examples.channel_nsch2d.problem import validate_config
    validate_config(target)
    before=json.loads(source.to_json());after=json.loads(target.to_json())
    for key in ('upper','lower','amplitude_min','amplitude_max'):
        after['geometry'][key]=before['geometry'][key]
    after['reference']['root']=before['reference']['root']
    after['operator']['validation_manifest']=before['operator']['validation_manifest']
    after['log']['run_name']=before['log']['run_name']
    if after!=before:
        raise ValueError('Geometry transfer permits only wall amplitudes/prior and artifact paths')
    a=(target.geometry.upper,target.geometry.lower)
    return dict(source_nominal=[source.geometry.upper,source.geometry.lower],
        target_nominal=list(a),source_prior=[source.geometry.amplitude_min,source.geometry.amplitude_max],
        target_prior=[target.geometry.amplitude_min,target.geometry.amplitude_max],
        inside_source_prior=all(source.geometry.amplitude_min<=v<=source.geometry.amplitude_max for v in a))


def audit_cold_geometry(run,source_run):
    """Compare cold state/time keys while retaining intentionally different geometries."""
    import numpy as np
    from ml_collections import ConfigDict
    run,source_run=Path(run),Path(source_run)
    configs=[ConfigDict(json.loads((p/'config.json').read_text()))for p in (source_run,run)]
    transfer=check_transfer(*configs)
    for p in (source_run,run):
        if json.loads((p/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed':
            raise ValueError('Both saved-run validations must pass before geometry comparison')
        if json.loads((p/'online-validation.json').read_text())['status']!='actual_rolling_inputs_reconstructed':
            raise ValueError('Both actual geometry input streams must be independently reconstructed')
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (run/name).read_bytes()!=(source_run/name).read_bytes():
            raise ValueError(f'Cold geometry control differs: {name}')
    with np.load(source_run/'sampling.npz',allow_pickle=False)as a,np.load(run/'sampling.npz',allow_pickle=False)as b:
        if set(a.files)!=set(b.files)or set(b.files)!={'indices','times','draw_ids','amplitudes'}:
            raise ValueError('Geometry controls require the explicit continuous-time online sampler')
        for name in ('indices','times','draw_ids'):np.testing.assert_array_equal(a[name],b[name])
        if np.array_equal(a['amplitudes'],b['amplitudes']):
            raise ValueError('Stronger-family amplitudes unexpectedly match the original prior')
        updates=len(b['times']);draws=b['draw_ids'].size
    for name in ('known_inputs.npz','known_inputs_final.npz'):
        with np.load(source_run/name,allow_pickle=False)as a,np.load(run/name,allow_pickle=False)as b:
            if set(a.files)!=set(b.files):raise ValueError('Geometry payload schemas differ')
            for key in a.files:
                if a[key].shape!=b[key].shape or a[key].dtype!=b[key].dtype:
                    raise ValueError(f'Geometry payload shape/dtype changed: {name}/{key}')
    return dict(status='cold_state_time_keys_and_geometry_only_change_passed',
        source_run=str(source_run),run=str(run),transfer=transfer,updates=updates,
        condition_draws=draws,initial_model_and_complete_optimizer_exact=True,
        actual_times_indices_and_online_draw_ids_exact=True,
        actual_each_family_geometry_inputs_independently_reconstructed=True,
        geometry_arrays_intentionally_different=True,
        scope='Matched cold method/budget/random draws, different geometry prior and independent nominal FEM target; no same-field or population comparison')


def evaluate(source_run,target_config,reference_process,output,platform):
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.problem import identity,walls,flow_rate,initial_field
    from examples.channel_nsch2d.data import geometry_data
    source_run,target_config,reference_process,output=map(Path,(source_run,target_config,reference_process,output))
    if any(not p.is_absolute()for p in (source_run,target_config,reference_process,output)):
        raise ValueError('Absolute source/config/process/output paths required')
    if output.exists():raise FileExistsError(output)
    owner=json.loads(reference_process.read_text())
    if owner['status']!='reference_and_input_screen_passed'or owner['active_child_pid']is not None:
        raise ValueError('Complete independent FEM/input validation must precede frozen evaluation')
    if Path('/proc/'+str(owner['runner_pid'])).exists():
        raise RuntimeError('FEM owner has not exited; keep environments mutually exclusive')
    source=ConfigDict(json.loads((source_run/'config.json').read_text()))
    c=ConfigDict(json.loads(target_config.read_text()));transfer=check_transfer(source,c)
    if json.loads((source_run/'summary.json').read_text())['status']!='completed':
        raise ValueError('A completed source model is required')
    if json.loads((source_run/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed':
        raise ValueError('The source model must pass independent saved-field replay')
    gate_path=Path(c.reference.root)/'reference-validation.json'
    gate=json.loads(gate_path.read_text())
    if gate['status']!='passed'or gate['identity']!=json.loads(json.dumps(identity(c))):
        raise ValueError('Matching refinement-validated local FEM is required')
    if platform=='cuda'and subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU is occupied; no overlapping frozen evaluation')
    os.environ['JAX_PLATFORMS']=platform
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.network import build_network,encode,fields
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    checkpoint=source_run/'model_final.eqx';original=checkpoint.read_bytes()
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(checkpoint,build_network(c,key))
    convert=lambda d:{k:jnp.asarray(v,dtype=v.dtype if v.dtype.kind in 'iu'else c.train.compute_dtype)for k,v in d.items()}
    predict=eqx.filter_jit(lambda model,d,x,t:fields(model,encode(model,d),x,t,d['amplitudes'],c))
    with np.load(source_run/'prediction.npz',allow_pickle=False)as f:old={k:f[k]for k in f.files}
    with np.load(source_run/'evaluation_inputs.npz',allow_pickle=False)as f:known=convert({k:f[k]for k in f.files})
    old_replay=np.asarray(predict(net,known,jnp.asarray(old['points'],dtype=c.train.compute_dtype),
        jnp.asarray(old['time'],dtype=c.train.compute_dtype)),dtype=np.float64)
    replay=float(np.max(np.abs(old_replay-old['fields'])))
    if not np.isfinite(old_replay).all()or replay>2e-5:
        raise ValueError(f'Target config changes source physical replay: {replay}')
    with np.load(gate['reference'],allow_pickle=False)as f:ref={k:f[k]for k in f.files}
    np.testing.assert_array_equal(ref['amplitudes'],np.asarray([c.geometry.upper,c.geometry.lower]))
    raw,operator=geometry_data(ref['amplitudes'],c);data=convert(raw)
    prediction=np.asarray(predict(net,data,jnp.asarray(ref['points'],dtype=c.train.compute_dtype),
        jnp.asarray(ref['time'],dtype=c.train.compute_dtype)),dtype=np.float64)
    if prediction.shape!=ref['fields'].shape or not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite or misaligned transferred physical fields')
    initial=np.asarray(initial_field(ref['points'],c,xp=np));ic=float(np.max(np.abs(prediction[0,:,0]-initial)))
    times=jnp.asarray([.5,3.,6.],dtype=c.train.compute_dtype)
    boundary=np.asarray(predict(net,data,jnp.asarray(raw['boundary'][:,:2],dtype=c.train.compute_dtype),times))
    wall=float(np.max(np.linalg.norm(boundary[:,raw['boundary'][:,4]==0,2:4],axis=-1)))
    z,w=np.polynomial.legendre.leggauss(64);x=np.linspace(0,c.geometry.length,17)
    bottom,top=walls(x,ref['amplitudes'],c,xp=np);height=top-bottom
    section=np.stack(np.broadcast_arrays(x[:,None],bottom[:,None]+height[:,None]*(z+1)/2),axis=-1)
    velocity=np.asarray(predict(net,data,jnp.asarray(section.reshape(-1,2),dtype=c.train.compute_dtype),times))[...,2]
    flux=np.sum(velocity.reshape(len(times),len(x),len(z))*w[None,None,:],axis=-1)*height[None,:]/2
    flux_gap=float(np.max(np.abs(flux-np.asarray(flow_rate(np.asarray(times),c,xp=np))[:,None])))
    if max(ic,wall,flux_gap)>2e-5:
        raise ValueError(f'Transferred hard constraints fail: IC={ic},wall={wall},flux={flux_gap}')
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    target=ref['fields'];metrics={}
    for name,sl,begin in (('phi',slice(0,1),0),('velocity',slice(2,4),1),('pressure',slice(4,5),1),('chemical',slice(1,2),1)):
        metrics[name]=relative(prediction[begin:,...,sl],target[begin:,...,sl])
    metrics['terminal_phi']=relative(prediction[-1,:,0],target[-1,:,0])
    metrics['dispersed_phase']=relative((1-prediction[...,0])/2,(1-target[...,0])/2)
    if checkpoint.read_bytes()!=original:raise ValueError('Frozen source model bytes changed')
    output.mkdir(parents=True)
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    np.savez_compressed(output/'prediction.npz',fields=prediction,points=ref['points'],
        time=ref['time'],reference=target,amplitudes=ref['amplitudes'])
    report=dict(status='frozen_geometry_transfer_completed',source_run=str(source_run),
        source_model=str(checkpoint),source_model_bytes_unchanged=True,
        source_saved_replay_gap=replay,transfer=transfer,reference=gate['reference'],
        reference_validation=str(gate_path),metrics=metrics,
        hard_constraints=dict(initial_phi_max_gap=ic,maximum_wall_speed=wall,maximum_flux_error=flux_gap),
        geometry_operator=operator,training_updates=0,
        scope='Frozen operator on one new nominal geometry; extrapolation if outside source prior. Independent localFEM reference; no fitted weights, labels or population claim')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k]for k in ('status','source_saved_replay_gap','transfer','metrics','hard_constraints')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','target-config','reference-process','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--platform',required=True,choices=('cpu','cuda'))
    a=p.parse_args();evaluate(a.source_run,a.target_config,a.reference_process,a.output,a.platform)
