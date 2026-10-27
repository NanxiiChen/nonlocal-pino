"""Identical-input gradient/SOAP/short-path repeatability probes; no production edits."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.losses import make_loss_fn
from examples.twophase2d.problem import load_reference,initial_field
from examples.twophase2d.train import _build_network,_make_step
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy

BASE=Path('/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-nonlocal')


def host(tree):
    return jax.tree.map(lambda a:np.asarray(a).copy() if eqx.is_array(a) else a,tree)


def difference(a,b):
    aa=jax.tree.leaves(eqx.filter(a,eqx.is_inexact_array));bb=jax.tree.leaves(eqx.filter(b,eqx.is_inexact_array))
    if len(aa)!=len(bb):raise ValueError('Tree mismatch')
    max_abs=0.;squared=0.;norm=0.;count=0
    for x,y in zip(aa,bb,strict=True):
        x=np.asarray(x);y=np.asarray(y)
        if x.shape!=y.shape:raise ValueError('Leaf shape mismatch')
        d=x-y;max_abs=max(max_abs,float(np.max(np.abs(d))))
        squared+=float(np.sum(d*d));norm+=float(np.sum(x*x));count+=int(np.count_nonzero(d))
    return dict(max_absolute=max_abs,relative_L2=float(np.sqrt(squared/norm)) if norm>0 else None,changed_entries=count)


def soap(state):
    result=state[1][0]
    if not all(hasattr(result,key) for key in ['GG','Q','exp_avg','exp_avg_sq']):raise TypeError('Unexpected upstream SOAP state')
    return result


class SymmetricAdjoint(eqx.Module):
    D: object

    def __matmul__(self,values):
        D=self.D
        @jax.custom_vjp
        def action(v):return D@v
        def forward(v):return D@v,None
        def backward(_,bar):return (D@bar,)
        action.defvjp(forward,backward)
        return action(values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=['ordinary','deterministic'],required=True)
    parser.add_argument('--adjoint',choices=['autodiff','symmetric'],required=True)
    parser.add_argument('--steps',type=int,required=True)
    args=parser.parse_args()
    if args.steps<2:raise ValueError('At least two steps')
    configure_gpu_policy(args.policy)
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='gpu':raise ValueError('GPU required')
    c=ConfigDict(json.loads((BASE/'config.json').read_text()))
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    ref=load_reference(c);points=jnp.asarray(ref['points']);mean=jnp.mean(initial_field(points,c,xp=jnp))
    kn,kt=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=_build_network(c,kn);initial=eqx.tree_deserialise_leaves(BASE/'model_initial.eqx',net)
    for a,b in zip(jax.tree.leaves(net),jax.tree.leaves(initial),strict=True):np.testing.assert_array_equal(a,b)
    times=[]
    for _ in range(args.steps):
        kt,key=jax.random.split(kt)
        margin=4*np.finfo(np.float32).eps*c.sample.Nt
        draw=jax.random.uniform(key,(c.sample.Nt,),minval=margin,maxval=1-margin,dtype=jnp.float32)
        times.append(((jnp.arange(c.sample.Nt)+draw)*c.physical.terminal_time/c.sample.Nt).astype(jnp.float64))
    out=ROOT/'reports/twophase-repeatability'/(args.policy+'-'+args.adjoint)
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'times.npy',np.asarray(jnp.stack(times)))
    D=PeriodicLaplacian(c.sample.Nx,c.geometry.length,c.nl.delta,dtype=np.float64)
    # Explicit adjoint oracle, independent symmetry/dot-product identity check.
    v=jnp.asarray(np.random.default_rng(1).normal(size=(4096,2)))
    w=jnp.asarray(np.random.default_rng(2).normal(size=(4096,2)))
    lhs=float(jnp.sum((D@v)*w));rhs=float(jnp.sum(v*(D@w)))
    np.testing.assert_allclose(lhs,rhs,rtol=2e-14,atol=1e-9)
    # Inspect the actual gather transpose independently from the neural network.
    pullback=jax.jit(lambda bar:jax.vjp(lambda value:D@value,v)[1](bar)[0])
    transpose=[host(pullback(w)) for _ in range(5)]
    exact=host(D@w)
    operator_records=[difference(transpose[0],a) for a in transpose[1:]]
    operator_oracle=difference(exact,transpose[0])
    apply=D if args.adjoint=='autodiff' else SymmetricAdjoint(D)
    loss=make_loss_fn(points,apply,c)
    grad_fn=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))
    begin=time.perf_counter();frozen=[]
    for _ in range(5):frozen.append(host(grad_fn(initial,times[0],mean)))
    frozen_seconds=time.perf_counter()-begin
    gradient_records=[difference(frozen[0][1],a[1]) for a in frozen[1:]]
    loss_values=[float(a[0][0]) for a in frozen]
    optimizer=make_optimizer(c);state0=optimizer.init(eqx.filter(initial,eqx.is_array))
    update=eqx.filter_jit(lambda gradient,state,model:optimizer.update(gradient,state,model))
    # Frozen gradient -> optimizer repetitions isolate internal eigensolver noise.
    same_gradient=[]
    for _ in range(3):same_gradient.append(host(update(frozen[0][1],state0,initial)))
    optimizer_records=[dict(updates=difference(same_gradient[0][0],x[0]),
                           GG=difference(soap(same_gradient[0][1]).GG,soap(x[1]).GG),
                           Q=difference(soap(same_gradient[0][1]).Q,soap(x[1]).Q)) for x in same_gradient[1:]]
    # Different repeated gradients differ only by numerical execution; compare
    # their SOAP initialization and second update with a COMMON second gradient.
    varied=[host(update(x[1],state0,initial)) for x in frozen]
    gradient2=grad_fn(initial,times[1],mean)[1]
    second=[host(update(gradient2,x[1],initial)) for x in varied]
    amplification=[dict(gradient=gradient_records[i-1],
        GG=difference(soap(varied[0][1]).GG,soap(varied[i][1]).GG),
        Q=difference(soap(varied[0][1]).Q,soap(varied[i][1]).Q),
        second_update=difference(second[0][0],second[i][0])) for i in range(1,len(frozen))]
    # Original canonical JIT step, restarted twice with identical immutable inputs.
    step=_make_step(loss,optimizer);paths=[];path_seconds=[];loss_paths=[]
    milestones=sorted(set([n for n in [1,2,5,10,20,50,100,args.steps] if n<=args.steps]))
    for rep in range(2):
        model=initial;state=state0;snapshots={};losses=[];start=time.perf_counter()
        for n,tt in enumerate(times,1):
            model,state,value,parts=step(model,state,tt,mean)
            value.block_until_ready();losses.append(float(value))
            if n in milestones:
                snapshots[n]=(host(model),host(soap(state).GG),host(soap(state).Q))
                print(json.dumps(dict(policy=args.policy,adjoint=args.adjoint,rep=rep,step=n,loss=losses[-1],elapsed=time.perf_counter()-start)),flush=True)
        paths.append(snapshots);path_seconds.append(time.perf_counter()-start);loss_paths.append(losses)
        eqx.tree_serialise_leaves(out/f'model_{rep}.eqx',model)
    np.savez_compressed(out/'loss_paths.npz',loss=np.asarray(loss_paths))
    path_records=[dict(step=n,parameters=difference(paths[0][n][0],paths[1][n][0]),
                       GG=difference(paths[0][n][1],paths[1][n][1]),
                       Q=difference(paths[0][n][2],paths[1][n][2]),
                       loss_absolute=abs(loss_paths[0][n-1]-loss_paths[1][n-1])) for n in milestones]
    row=dict(status='completed',policy=args.policy,adjoint=args.adjoint,steps=args.steps,
             exact_training_initial_arrays=True,operator_dot_product_absolute=abs(lhs-rhs),
             operator_transpose_repeat=operator_records,operator_transpose_oracle=operator_oracle,
             frozen_loss=loss_values,frozen_gradient_repeat=gradient_records,
             frozen_gradient_seconds_including_compile=frozen_seconds,
             optimizer_identical_gradient_repeat=optimizer_records,
             gradient_to_soap_amplification=amplification,
             path_seconds=path_seconds,path_repeat=path_records,
             reference_field_values_in_loss=False,source_run=str(BASE))
    (out/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=str(out)),indent=2),flush=True)


if __name__=='__main__':main()
