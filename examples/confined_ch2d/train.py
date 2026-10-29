"""Canonical cold start for confined CH; nonlocal or local AD spatial operators."""
from pathlib import Path
import sys
if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from absl import app,flags
from ml_collections import config_flags
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import json,shutil,time
from datetime import datetime,timezone
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import with_run_seed,configure_gpu_policy
from examples.twophase2d.refinement import refine
from examples.confined_ch2d.problem import validate_config,load_reference,initial_field,boundary_samples,relative
from examples.confined_ch2d.network import PhaseNetwork,predict_fields
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.sampling import temporal_samples

_CONFIG=config_flags.DEFINE_config_file('config',str(Path(__file__).parent/'configs/train_coarsening_windows.py'))
_SEED=flags.DEFINE_integer('seed',None,'Run provenance, not a strategy parameter')


def _build_network(c,key):
    from examples.confined_ch2d.network import GlobalBasisNetwork,GlobalMixedNetwork,TanhPhaseNetwork
    if 'phase' in c and 'output_map' in c.phase:constructor=TanhPhaseNetwork
    elif 'model' in c.arch and c.arch.model=='global_mixed':constructor=GlobalMixedNetwork
    else:constructor=GlobalBasisNetwork if 'model' in c.arch and c.arch.model=='global_basis' else PhaseNetwork
    with jax.enable_x64(False):net=constructor(c,key)
    return jax.tree.map(lambda v:v.astype(jnp.float64) if eqx.is_inexact_array(v) else v,net)


def _make_step(loss,optimizer):
    @eqx.filter_jit
    def step(net,state,times,mean):
        (value,parts),g=eqx.filter_value_and_grad(loss,has_aux=True)(net,times,mean)
        updates,state=optimizer.update(g,state,net)
        return eqx.apply_updates(net,updates),state,value,parts
    return step


def train(c):
    validate_config(c);configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    if 'windows' in c.train:
        from examples.confined_ch2d.window_training import train_windows
        if not c.log.run_name:
            from ml_collections import ConfigDict
            c=ConfigDict(c.to_dict())
            c.log.run_name=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-nonlocal-windows'
        return train_windows(c)
    ref=load_reference(c);x=jnp.asarray(ref['points']);volumes=jnp.asarray(ref['volumes'])
    mean=jnp.dot(volumes,initial_field(x,c,xp=jnp))/jnp.sum(volumes)
    kn,kt=jax.random.split(jax.random.PRNGKey(c.train.seed));net=_build_network(c,kn)
    D=RegionalLaplacian(ref['points'],ref['volumes'],c) if c.nl.method=='nonlocal' else None
    boundary,normals,bweights,_=boundary_samples(c)
    loss=make_loss_fn(x,volumes,D,jnp.asarray(boundary),jnp.asarray(normals),jnp.asarray(bweights),c)
    opt=make_optimizer(c);state=opt.init(eqx.filter(net,eqx.is_array));step=_make_step(loss,opt)
    balanced='loss_balance' in c.train
    if balanced:
        from examples.confined_ch2d.loss_balancing import make_balanced_step,frozen_joint_loss
        step=make_balanced_step(loss,opt,c)
        balance_weights=jnp.ones(2,dtype=x.dtype);balance_norms=jnp.zeros(2,dtype=x.dtype)
        balance_counts=dict(phase_only=0,chemical_only=0,joint=0)
    name=c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-'+c.nl.method
    run=Path(c.log.log_dir)/name;run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(c.to_json(indent=2)+'\n')
    root=Path(__file__).resolve().parents[2]
    shutil.copytree(Path(__file__).parent,run/'source/examples/confined_ch2d',ignore=shutil.ignore_patterns('__pycache__'))
    for rel in ('examples/twophase2d/refinement.py','src/nonlocal_pinn/training/optimizers.py',
        'src/nonlocal_pinn/training/runtime.py','src/nonlocal_pinn/nn/arch.py','src/nonlocal_pinn/nn/embedding.py'):
        target=run/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
    eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    target='nonlocal' if D is not None else 'local'
    @eqx.filter_jit
    def evaluate(model):return jax.vmap(lambda t:predict_fields(model,x,t,mean,volumes,D))(jnp.asarray(ref['time']))
    start=time.perf_counter();timing=[];best=np.inf;time_trace=[];horizon_trace=[]
    def assess(completed):
        nonlocal best
        values=np.asarray(evaluate(net));fields=dict(phi=values[...,0],mu=values[...,1])
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite prediction')
        record=dict(step=completed,elapsed_seconds=time.perf_counter()-start)
        for key,values in fields.items():
            for kind in ('local','nonlocal'):
                b=ref[key+'_'+kind]
                record['pooled_rel_l2_'+key+'_'+kind]=relative(values,b,ref['volumes'])
                record['terminal_rel_l2_'+key+'_'+kind]=relative(values[-1],b[-1],ref['volumes'])
                record['per_time_rel_l2_'+key+'_'+kind]=[relative(a,bb,ref['volumes']) for a,bb in zip(values,b,strict=True)]
            record['pooled_rel_l2_'+key]=record['pooled_rel_l2_'+key+'_'+target]
        record['mass_max_drift']=float(np.max(np.abs(fields['phi']@ref['volumes']/ref['volumes'].sum()-float(mean))))
        record['phase_min']=float(fields['phi'].min());record['phase_max']=float(fields['phi'].max())
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if record['pooled_rel_l2_phi_local']<best:
            best=record['pooled_rel_l2_phi_local'];eqx.tree_serialise_leaves(run/'model_best.eqx',net)
            np.savez_compressed(run/'prediction_best.npz',time=ref['time'],points=ref['points'],volumes=ref['volumes'],**fields)
            (run/'best_metrics.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(f'[{completed}] phiL2local={record["pooled_rel_l2_phi_local"]:.5g}, native={record["pooled_rel_l2_phi"]:.5g}',flush=True)
        return record,fields
    metrics,fields=assess(0)
    with (run/'loss.jsonl').open('x') as log:
        for i in range(1,c.train.iterations+1):
            kt,key=jax.random.split(kt)
            times,horizon=temporal_samples(c,i,key)
            if 'temporal' in c.sample:
                time_trace.append(np.asarray(times));horizon_trace.append(horizon)
            before=time.perf_counter()
            if balanced:
                net,state,balance_weights,balance_norms,value,parts,effective,active=step(
                    net,state,balance_weights,balance_norms,times,mean,jnp.asarray(i-1))
                active_host=np.asarray(active)
                stage='joint' if active_host.all() else ('phase_only' if active_host[0] else 'chemical_only')
                balance_counts[stage]+=1
            else:net,state,value,parts=step(net,state,times,mean)
            value.block_until_ready();timing.append(time.perf_counter()-before)
            if not np.isfinite(float(value)):raise FloatingPointError(f'Nonfinite loss at{i}')
            if i==1 or i%100==0:
                record=dict(step=i,total=float(value),parts=np.asarray(parts).tolist(),training_horizon=horizon)
                if balanced:
                    record.update(weights=np.asarray(balance_weights).tolist(),effective_weights=np.asarray(effective).tolist(),
                        active=active_host.tolist(),stage=stage,
                        gradient_norms=np.asarray(balance_norms).tolist() if c.train.loss_balance.kind=='gradient_norm' else None)
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if balanced:
                settings=c.train.loss_balance
                refresh=settings.kind=='gradient_norm' and (i-1)%settings.update_every==0
                switch=settings.kind=='stagger' and (i-1)%settings.switch_every==0
                if i==1 or refresh or switch or i==c.train.iterations:
                    with (run/'balancing.jsonl').open('a') as file:
                        file.write(json.dumps(dict(step=i,parts=np.asarray(parts).tolist(),weights=np.asarray(balance_weights).tolist(),
                            effective_weights=np.asarray(effective).tolist(),active=active_host.tolist(),stage=stage,
                            refreshed=refresh,gradient_norms=np.asarray(balance_norms).tolist() if settings.kind=='gradient_norm' else None),allow_nan=False)+'\n')
            if i%c.log.eval_every==0 or i==c.train.iterations:metrics,fields=assess(i)
            if i%c.log.checkpoint_every==0:eqx.tree_serialise_leaves(run/f'ckpt_{i:06d}.eqx',net)
    if 'temporal' in c.sample:
        np.savez_compressed(run/'training_times.npz',times=np.stack(time_trace),horizons=np.asarray(horizon_trace))
    eqx.tree_serialise_leaves(run/'model_soap.eqx',net)
    def callback(candidate,iteration):
        nonlocal net,metrics,fields
        net=candidate;metrics,fields=assess(c.train.iterations+iteration)
    refinement_function=refine
    refinement_loss=loss
    if 'causal' in c.loss:
        from ml_collections import ConfigDict
        uniform=ConfigDict(c.to_dict());del uniform.loss.causal
        refinement_loss=make_loss_fn(x,volumes,D,jnp.asarray(boundary),jnp.asarray(normals),jnp.asarray(bweights),uniform)
    if balanced:
        refinement_loss=frozen_joint_loss(refinement_loss,balance_weights,c)
        balance_record=dict(kind=c.train.loss_balance.kind,soap_update_counts=balance_counts,
            final_weights=np.asarray(balance_weights).tolist(),refinement=c.train.loss_balance.refinement,
            refinement_effective_weights=(np.asarray(balance_weights)*np.asarray([c.loss.phase_weight,c.loss.chemical_weight])).tolist(),
            shared_parameters='All trainable parameters update; stagger alternates losses, not frozen fields')
        (run/'balancing_summary.json').write_text(json.dumps(balance_record,indent=2,allow_nan=False)+'\n')
    if 'optimizer' in c.train.refinement and c.train.refinement.optimizer=='newton_cg':
        from examples.confined_ch2d.newton_refinement import refine as refinement_function
    net,refinement=refinement_function(net,refinement_loss,mean,c,run,callback)
    if balanced:
        refinement['loss_balance']=balance_record
        (run/'refinement_summary.json').write_text(json.dumps(refinement,indent=2,allow_nan=False)+'\n')
    if 'causal' in c.loss:
        refinement['causal_weights']='disabled; uniform fixed full-horizon objective'
        (run/'refinement_summary.json').write_text(json.dumps(refinement,indent=2,allow_nan=False)+'\n')
    metrics,fields=assess(c.train.iterations+refinement['iterations'])
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    np.savez_compressed(run/'prediction.npz',time=ref['time'],points=ref['points'],volumes=ref['volumes'],**fields)
    summary=dict(cold_start=True,method=c.nl.method,wall_seconds=time.perf_counter()-start,
        first_update_seconds=timing[0],mean_update_seconds_after_100=float(np.mean(timing[100:])) if len(timing)>100 else None,
        refinement=refinement,final_metrics=metrics,quadrature_points=len(x),
        reference_path=c.reference.path,best_checkpoint='model_best.eqx',
        best_selection='minimum evaluated pooled common-local phase L2; no reference labels train')
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2),flush=True)
    return run


def _main(_):train(with_run_seed(_CONFIG.value,_SEED.value))


if __name__=='__main__':app.run(_main)
