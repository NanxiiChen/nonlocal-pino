"""Full local KKS controls dispatched by corrosion2d/train.py."""
import json
import os
from pathlib import Path
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.local_ad import local_loss_factory
from examples.corrosion2d.local_reference import load as load_local_reference
from examples.corrosion2d.residuals import validate_normalized_data,validate_transport_options
from examples.corrosion2d.training_precision import computation_dtype,cast_floating,audit_floating
from nonlocal_pinn.operators import build_D


def train_local(config):
    if config.experiment.case!='corrosion2d':raise ValueError('Local control currently supports corrosion2d only')
    if config.experiment.formulation!='phase':raise ValueError('Local control requires original phase residual')
    if config.experiment.data_loss_mode!='s_bc' or config.experiment.transport_evaluation!='kks_raw_edges':
        raise ValueError('Local control requires explicit KKS/s_bc/raw-q strategy')
    if config.experiment.initial_sampling!='matched_graph' or config.experiment.boundary_sampling!='matched_graph':
        raise ValueError('Local control requires the explicitly matched nodal IC/BC data')
    if config.experiment.ch_norm!='l1' or config.loss.stagger_steps or config.loss.t_ramp_steps or config.experiment.ch_rms_after or config.experiment.lr_tail_start:
        raise ValueError('Local control requires joint/full-horizon/L1/no-continuation configuration')
    if len(config.loss.gradnorm_bias)!=6:raise ValueError('Local control requires six explicit loss biases, including no flux')
    r.validate_loss_schedule(config);r.validate_time_window(config)
    validate_normalized_data(config);validate_transport_options(config)
    bias=np.asarray(config.loss.gradnorm_bias)
    if not np.isfinite(bias).all() or np.any(bias<=0):raise ValueError('Local loss biases must be finite and positive')
    if not config.log.run_name:raise ValueError('Local research cold start requires a fresh explicit run_name')
    dtype=computation_dtype(config)
    jax.config.update('jax_enable_x64',dtype==np.dtype('float64'))
    jax.config.update('jax_default_matmul_precision',config.train.matmul_precision)
    reference,reference_metadata=load_local_reference(config)
    run=Path(config.log.log_dir)/config.log.run_name
    run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(config.to_json(indent=2)+'\n')
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),Path(__file__).with_name('local_ad.py'),Path(__file__).with_name('local_reference.py'),
           Path(__file__).with_name('residual_experiment.py'),Path(__file__).with_name('residuals.py'),
           Path(__file__).with_name('training_precision.py'),Path(__file__).with_name('training_entry.py'),
           Path(__file__).with_name('training_data.py'),Path(__file__).with_name('transport_optimizer.py'),
           Path(__file__).with_name('train.py'),root/'src/nonlocal_pinn/nn/arch.py',
           root/'src/nonlocal_pinn/nn/constraints.py',root/'src/nonlocal_pinn/training/runtime.py',
           *sorted((Path(__file__).parent/'configs').glob('*.py'))]
    for source in files:
        target=run/'source'/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    (run/'source/files.json').write_text(json.dumps([str(p.relative_to(root)) for p in files],indent=2)+'\n')
    kind={'local_ad_tensor':'local_tensor','local_ad_joint':'local_joint'}[config.experiment.spatial_operator]
    with jax.enable_x64(False):
        xx,yy,points=r.grid(config)
        x_grid=jnp.asarray(points,dtype=jnp.float32)
        zero,one,_=r.boundary_values(config,points)
        neu=np.isclose(points[:,1],0)&~zero&~one
        if int(neu.sum())!=90:raise ValueError(f'Unexpected local Neumann node count: {neu.sum()}')
        x_neumann=x_grid[neu]
        x_zero,x_one=x_grid[zero],x_grid[one]
        x_ic,ic=r.training_initial_data(config,points,points)
        expected=np.stack([reference['phi'][0].ravel(),reference['c'][0].ravel()],-1)
        if not np.allclose(np.asarray(ic),expected,rtol=0,atol=3e-7):
            raise ValueError('Local initial targets differ from independent reference')
        key_net,key_time=jax.random.split(jax.random.PRNGKey(config.train.seed))
        net=r.build_model(config,key_net)
        op=build_D(x_grid,config.nl.delta,method='sparse',kernel=config.nl.kernel)
        pool=op.build_subgraph_pool(config.sample.K_core,config.sample.K_pool)
        batches=[];counts=[]
        for ids,_,core,_ in pool:
            valid=np.asarray(core)&(~(zero|one))[np.asarray(ids)]
            count=int(valid.sum());counts.append(count)
            selected=np.asarray(ids)[np.flatnonzero(valid)]
            selected=np.pad(selected,(0,config.sample.K_core-count),mode='edge')
            mask=jnp.arange(config.sample.K_core)<count
            if kind=='local_tensor':batches.append((x_grid[selected],None,mask))
            else:batches.append((jnp.zeros((config.sample.K_core*config.sample.Nt,3),dtype=jnp.float32),None,jnp.tile(mask,config.sample.Nt)))
        del pool,op
    net,x_grid,x_ic,ic,x_zero,x_one,x_neumann,batches=cast_floating((net,x_grid,x_ic,ic,x_zero,x_one,x_neumann,batches),dtype)
    eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    loss,_=local_loss_factory(config,r.nondim(config),x_ic,ic,x_zero,x_one,x_neumann,kind)
    optimizer=r.make_tail_optimizer(config);state=optimizer.init(eqx.filter(net,eqx.is_array))
    step=r.make_step(loss,optimizer,config)
    weights=jnp.ones(6,dtype=dtype);active=jnp.ones(6,dtype=bool)
    times_eval=jnp.asarray(np.asarray(reference['time'],dtype=np.float32),dtype=dtype)
    cle=config.physical.c_le;dc=config.physical.c_se-cle
    target_p,target_c=reference['phi'],reference['c']
    target_q=target_c-dc*(3*target_p**2-2*target_p**3)-cle
    free=~(zero|one)
    writer=SummaryWriter(str(run));start=time.perf_counter()
    precision={'compute_dtype':str(dtype),'parameters':audit_floating(net,dtype),
               'optimizer_initial':audit_floating(state,dtype),
               'data':audit_floating((x_grid,x_ic,ic,x_zero,x_one,x_neumann,batches),dtype),
               'input_policy':'Common FP32 initial/data/time/pair draws promoted exactly'}
    (run/'precision.json').write_text(json.dumps(precision,indent=2)+'\n')

    @eqx.filter_jit
    def predict(model):
        raw=jax.vmap(lambda t:jax.vmap(model.net,in_axes=(0,None))(x_grid,t))(times_eval)
        fields=model.constraint(raw)
        q=.5*cle*(jnp.tanh(raw[...,1])-1.)
        return fields,q

    @eqx.filter_jit
    def no_flux(model):
        from examples.corrosion2d.residuals import normalized_data_fields
        direction=jnp.zeros_like(x_neumann).at[:,1].set(1.)
        return jax.jvp(lambda x:normalized_data_fields(model,x,times_eval),(x_neumann,),(direction,))[1]

    def evaluate(completed):
        fields,q=predict(net)
        prediction=np.asarray(fields).reshape(21,*xx.shape,2)
        q=np.asarray(q,dtype=np.float64).reshape(21,*xx.shape)
        if not np.isfinite(prediction).all() or not np.isfinite(q).all():raise FloatingPointError('Nonfinite local prediction')
        metrics=r.field_metrics(prediction,reference)
        metrics['terminal_rel_l2_q_free']=float(np.linalg.norm((q[-1]-target_q[-1]).ravel()[free])/np.linalg.norm(target_q[-1].ravel()[free]))
        metrics['terminal_rel_l2_q']=float(np.linalg.norm(q[-1]-target_q[-1])/np.linalg.norm(target_q[-1]))
        metrics['pooled_rel_l2_q']=float(np.linalg.norm(q-target_q)/np.linalg.norm(target_q))
        metrics['boundary_max_abs_error']=r.boundary_error(prediction,zero,one)
        normal=np.asarray(no_flux(net),dtype=np.float64)
        metrics['neumann_phi_rms']=float(np.sqrt(np.mean(normal[...,0]**2)))
        metrics['neumann_q_rms_over_cle']=float(np.sqrt(np.mean(normal[...,1]**2)))
        metrics['step']=completed;metrics['elapsed_seconds']=time.perf_counter()-start
        with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        for key,value in metrics.items():
            if isinstance(value,(int,float)):writer.add_scalar('eval/'+key,value,completed)
        print(f'[{kind} {completed}] terminal phi L2={metrics["terminal_rel_l2_phi"]:.7g}, c L2={metrics["terminal_rel_l2_c"]:.7g}, '
              f'q L2={metrics["terminal_rel_l2_q_free"]:.7g}, no-flux phi/s RMS={metrics["neumann_phi_rms"]:.3g}/{metrics["neumann_q_rms_over_cle"]:.3g} '
              f'({metrics["elapsed_seconds"]:.1f}s)',flush=True)
        return metrics,prediction

    evaluate(0)
    labels=('ac','ch','bc_zero','bc_one','ic','no_flux')
    for i in range(config.train.iterations):
        with jax.enable_x64(False):
            times=r.sample_training_times(config,key_time,i)
            if kind=='local_joint':
                xt=jax.random.uniform(jax.random.fold_in(key_time,100000+i),(config.sample.K_core*config.sample.Nt,3),dtype=jnp.float32)
                xt=xt*jnp.asarray([1.,.5,1.-config.sample.t_min],dtype=jnp.float32)+jnp.asarray([0.,0.,config.sample.t_min],dtype=jnp.float32)
        x,sub,mask=batches[i%len(batches)]
        if kind=='local_joint':x=xt.astype(dtype)
        net,state,weights,losses,finite=step(net,state,weights,times.astype(dtype),x,sub,mask,
                                          jnp.asarray(i%config.train.gradnorm_update_every==0),active)
        if not bool(finite):raise FloatingPointError(f'Local nonfinite at {i+1}: {losses}')
        completed=i+1
        if completed%config.log.log_every==0 or completed==1:
            record={'step':completed,'losses':np.asarray(losses).tolist(),'adaptive_weights':np.asarray(weights).tolist(),
                    'weights':(np.asarray(weights)*np.asarray(config.loss.gradnorm_bias)).tolist(),'active':[True]*6,
                    'labels':labels,'stage':'joint','update_weights':i%config.train.gradnorm_update_every==0}
            with (run/'loss_history.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(f'update {completed}: losses={record["losses"]}, weights={record["weights"]}',flush=True)
        if completed%config.log.eval_every==0 or completed==config.train.iterations:
            final,prediction=evaluate(completed)
        if completed%config.experiment.checkpoint_every==0:
            eqx.tree_serialise_leaves(run/f'ckpt_{completed:06d}.eqx',net)
    precision['parameters_final']=audit_floating(net,dtype);precision['optimizer_final']=audit_floating(state,dtype)
    (run/'precision.json').write_text(json.dumps(precision,indent=2)+'\n')
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    np.savez_compressed(run/'prediction.npz',X=xx,Y=yy,time=reference['time'],phi=prediction[...,0],c=prediction[...,1])
    summary={'run':str(run),'updates':config.train.iterations,'physical_seconds':config.nondim.T_c,
             'model_path':'model_final.eqx','reference':config.experiment.local_reference_path,
             'spatial_operator':config.experiment.spatial_operator,'sampling':kind,
             'Neumann':'bottom outside mouth: phi_n=0,q_n/c_le=0; sixth GradNorm loss',
             'Neumann_nodes':int(neu.sum()),'active_PDE_pairs_min_max':[min(counts)*config.sample.Nt,max(counts)*config.sample.Nt],
             'reference_refinement':reference_metadata['refinement'],'final_metrics':final,
             'xla_flags':os.environ['XLA_FLAGS'],'wall_seconds':time.perf_counter()-start}
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');writer.close()
