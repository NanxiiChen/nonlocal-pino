"""SUPERVISED representation diagnostic only. Never initialize a PINN from it."""
from pathlib import Path
import sys,json,time,shutil
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import jax,jax.numpy as jnp,equinox as eqx
from nonlocal_pinn.training.runtime import with_run_seed,configure_gpu_policy
from nonlocal_pinn.training.optimizers import make_optimizer
from examples.confined_ch2d.configs.train_coarsening_global_separate import get_config
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.train import _build_network,_make_step
from examples.confined_ch2d.network import predict_fields
c=with_run_seed(get_config(),42);configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
run=Path(c.log.log_dir)/'20261028-chmixed-representation-01';run.mkdir(exist_ok=False)
(run/'scope.txt').write_text('SUPERVISED representation audit, not a physics-only PINN result. Reference targets train this diagnostic only. Never use these weights to initialize a PINN.\n')
(run/'config.json').write_text(c.to_json(indent=2)+'\n');shutil.copyfile(__file__,run/'diagnostic.py')
shutil.copytree(ROOT/'examples/confined_ch2d',run/'source/examples/confined_ch2d',ignore=shutil.ignore_patterns('__pycache__'))
ref=load_reference(c);v=ref['volumes'];p=c.physical;A=native_matrix(ref['points'],v,c);A2=A@A;phi0=initial_field(ref['points'],c,xp=np)
nodes=np.r_[0.,c.sample.temporal.nodes];times=np.unique(np.r_[nodes,ref['time']])
rhs=lambda t,phi:-p.mobility*(A@(p.beta*(phi**3-phi))+p.epsilon**2*(A2@phi))
jac=lambda t,phi:-p.mobility*(A@sparse.diags(p.beta*(3*phi**2-1))+p.epsilon**2*A2)
sol=solve_ivp(rhs,(0,p.terminal_time),phi0,method='BDF',jac=jac,rtol=1e-9,atol=1e-11,max_step=c.reference.refined_dt,t_eval=times)
if not sol.success:raise RuntimeError(sol.message)
phi=sol.y.T;mu=p.beta*(phi**3-phi)+p.epsilon**2*(A@phi.T).T
replay=relative(phi[np.searchsorted(times,ref['time'])],ref['phi_nonlocal'],v)
if replay>1e-5:raise ValueError('Native reference validation failed')
indices=np.searchsorted(times,nodes);target=np.stack([phi[indices],mu[indices]],axis=-1)
np.savez_compressed(run/'targets_supervised_only.npz',times=nodes,fields=target)
points=jnp.asarray(ref['points']);vol=jnp.asarray(v);mean=jnp.dot(vol,phi0)/vol.sum();targets=jnp.asarray(target)
key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));net=_build_network(c,key)
eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
if (run/'model_initial.eqx').read_bytes()!=(Path(c.log.log_dir)/'20261028-chmixed-separate-01/model_initial.eqx').read_bytes():raise ValueError('Diagnostic initial parameters differ from PDE control')
@eqx.filter_jit
def evaluate(model):return jax.vmap(lambda t:predict_fields(model,points,t,mean,vol))(jnp.asarray(ref['time']))
def loss(model,unused_times,initial_mean):
    values=jax.vmap(lambda t:predict_fields(model,points,t,initial_mean,vol))(jnp.asarray(nodes))
    error=(values-targets)/jnp.asarray([1.,c.arch.chemical_scale])
    parts=jnp.mean(jnp.sum(vol[None,:,None]*error**2,axis=1)/vol.sum(),axis=0)
    return jnp.sum(parts),parts
opt=make_optimizer(c);state=opt.init(eqx.filter(net,eqx.is_array));step=_make_step(loss,opt)
records=[];start=time.perf_counter()
for i in range(1,c.train.iterations+1):
    net,state,value,parts=step(net,state,jnp.asarray(c.sample.temporal.nodes),mean);value.block_until_ready()
    if not np.isfinite(float(value)):raise FloatingPointError('Nonfinite supervised diagnostic')
    if i==1 or i%250==0:
        actual=np.asarray(evaluate(net));row=dict(step=i,loss=float(value),parts=np.asarray(parts).tolist(),elapsed_seconds=time.perf_counter()-start)
        for f,name in enumerate(('phi','mu')):
            row[name+'_native']=relative(actual[...,f],ref[name+'_nonlocal'],v)
            row[name+'_terminal_native']=relative(actual[-1,:,f],ref[name+'_nonlocal'][-1],v)
        records.append(row);(run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));print(row,flush=True)
eqx.tree_serialise_leaves(run/'supervised_model.eqx',net)
actual=np.asarray(evaluate(net));np.savez_compressed(run/'supervised_prediction.npz',phi=actual[...,0],mu=actual[...,1])
report=dict(scope='Supervised field representation audit ONLY; no PDE training/accuracy claim or transfer of weights.',native_reference_replay=replay,budget='5000SOAP, no L-BFGS',final=records[-1],run=str(run))
(run/'summary.json').write_text(json.dumps(report,indent=2)+'\n');Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
