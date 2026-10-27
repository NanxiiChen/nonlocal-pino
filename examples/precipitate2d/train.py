"""Canonical cold-start entry: phase PINN plus differentiable elastic equilibrium."""
from pathlib import Path
import sys
if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from absl import app,flags
from ml_collections import config_flags
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import json
import shutil
import time
from datetime import datetime,timezone
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import with_run_seed,configure_gpu_policy
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.refinement import refine
from examples.precipitate2d.problem import validate_config,load_reference,initial_field
from examples.precipitate2d.network import PhaseNetwork,predict_fields
from examples.precipitate2d.mechanics import ElasticEquilibrium
from examples.precipitate2d.losses import make_loss_fn

_CONFIG = config_flags.DEFINE_config_file('config',str(Path(__file__).parent/'configs/train_base.py'))
_SEED = flags.DEFINE_integer('seed',None,'Run provenance, not a tuned strategy parameter')


def _build_network(c,key):
    with jax.enable_x64(False):net = PhaseNetwork(c,key)
    return jax.tree.map(lambda a:a.astype(jnp.float64) if eqx.is_inexact_array(a) else a,net)


def _make_step(loss,optimizer):
    @eqx.filter_jit
    def step(net,state,times,mean):
        (value,parts),g = eqx.filter_value_and_grad(loss,has_aux=True)(net,times,mean)
        updates,state = optimizer.update(g,state,net)
        return eqx.apply_updates(net,updates),state,value,parts
    return step


def train(c):
    validate_config(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    ref = load_reference(c)
    x = jnp.asarray(ref['points'])
    mean = jnp.mean(initial_field(x,c,xp=jnp))
    kn,kt = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = _build_network(c,kn)
    D = PeriodicLaplacian(c.sample.Nx,c.geometry.length,c.nl.delta,dtype=np.float64) if c.nl.method=='nonlocal' else None
    mechanics = ElasticEquilibrium(c)
    loss = make_loss_fn(x,D,mechanics,c)
    opt = make_optimizer(c)
    state = opt.init(eqx.filter(net,eqx.is_array))
    step = _make_step(loss,opt)
    name = c.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-'+c.nl.method
    run = Path(c.log.log_dir)/name
    run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(c.to_json(indent=2)+'\n')
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(Path(__file__).parent,run/'source/examples/precipitate2d',ignore=shutil.ignore_patterns('__pycache__'))
    for rel in ('examples/twophase2d/operators.py','examples/twophase2d/refinement.py',
        'src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py',
        'src/nonlocal_pinn/nn/arch.py','src/nonlocal_pinn/nn/embedding.py'):
        target = run/'source'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/rel,target)
    eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
    target = 'nonlocal' if D is not None else 'local'
    times_eval = jnp.asarray(ref['time'])
    @eqx.filter_jit
    def evaluate(model):
        def at(t):
            fields = predict_fields(model,x,t,mean)
            _,u,s = mechanics(fields[:,0])
            return fields,u,s
        return jax.vmap(at)(times_eval)
    start = time.perf_counter()
    timing = []
    best = np.inf
    def assess(completed):
        nonlocal best
        pq,u,s = (np.asarray(a) for a in evaluate(net))
        n,nt = c.sample.Nx,len(times_eval)
        fields = dict(phi=pq[...,0].reshape(nt,n,n),q=pq[...,1].reshape(nt,n,n),
                      u=u.reshape(nt,n,n,2),stress=s.reshape(nt,n,n,3))
        if not all(np.isfinite(a).all() for a in fields.values()):raise FloatingPointError('Nonfinite physical fields')
        relative = lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        record = dict(step=completed,elapsed_seconds=time.perf_counter()-start)
        for key,a in fields.items():
            b = ref[key+'_'+target]
            record['pooled_rel_l2_'+key] = relative(a,b)
            record['terminal_rel_l2_'+key] = relative(a[-1],b[-1])
            record['per_time_rel_l2_'+key] = [relative(aa,bb) for aa,bb in zip(a,b,strict=True)]
        record['pooled_rel_l2_phi_local'] = relative(fields['phi'],ref['phi_local'])
        record['phase_mass_max_drift'] = float(np.max(np.abs(fields['phi'].mean(axis=(1,2))-float(mean))))
        record['phase_min'],record['phase_max'] = float(fields['phi'].min()),float(fields['phi'].max())
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if record['pooled_rel_l2_phi']<best:
            best = record['pooled_rel_l2_phi']
            eqx.tree_serialise_leaves(run/'model_best.eqx',net)
            np.savez_compressed(run/'prediction_best.npz',time=ref['time'],points=ref['points'],**fields)
            (run/'best_metrics.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(f'[{completed}] phiL2={record["pooled_rel_l2_phi"]:.5g}, uL2={record["pooled_rel_l2_u"]:.5g}',flush=True)
        return record,fields
    metrics,fields = assess(0)
    with (run/'loss.jsonl').open('x') as log:
        for i in range(1,c.train.iterations+1):
            kt,key = jax.random.split(kt)
            draws = jax.random.uniform(key,(c.sample.Nt,),dtype=jnp.float64,minval=1e-6,maxval=1-1e-6)
            times = (jnp.arange(c.sample.Nt)+draws)*c.physical.terminal_time/c.sample.Nt
            before = time.perf_counter()
            net,state,value,parts = step(net,state,times,mean)
            value.block_until_ready()
            timing.append(time.perf_counter()-before)
            if not np.isfinite(float(value)):raise FloatingPointError(f'Loss divergence at {i}')
            if i==1 or i%100==0:
                log.write(json.dumps(dict(step=i,total=float(value),parts=np.asarray(parts).tolist()),allow_nan=False)+'\n');log.flush()
            if i%c.log.eval_every==0 or i==c.train.iterations:metrics,fields = assess(i)
            if i%c.log.checkpoint_every==0:eqx.tree_serialise_leaves(run/f'ckpt_{i:06d}.eqx',net)
    eqx.tree_serialise_leaves(run/'model_soap.eqx',net)
    def callback(candidate,iteration):
        nonlocal net,metrics,fields
        net = candidate
        metrics,fields = assess(c.train.iterations+iteration)
    net,refinement = refine(net,loss,mean,c,run,callback)
    metrics,fields = assess(c.train.iterations+refinement['iterations'])
    eqx.tree_serialise_leaves(run/'model_final.eqx',net)
    np.savez_compressed(run/'prediction.npz',time=ref['time'],points=ref['points'],**fields)
    summary = dict(cold_start=True,method=c.nl.method,mechanics='exact differentiable spectral equilibrium',
        wall_seconds=time.perf_counter()-start,first_update_seconds=timing[0],
        mean_update_seconds_after_100=float(np.mean(timing[100:])) if len(timing)>100 else None,
        refinement=refinement,final_metrics=metrics,reference_path=c.reference.path,
        best_checkpoint='model_best.eqx',best_selection='minimum evaluated pooled native phase relative L2; validation only, not a training target')
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return run


def _main(_):
    train(with_run_seed(_CONFIG.value,_SEED.value))


if __name__=='__main__':app.run(_main)
