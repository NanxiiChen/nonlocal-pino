"""Fresh-process GPU update probe with exact recorded cold inputs; no training labels."""
import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--repeats',type=int,required=True)
    parser.add_argument('--gpu-policy',choices=('ordinary','deterministic'),required=True)
    parser.add_argument('--fixed-adjoint',action='store_true')
    args=parser.parse_args()
    if args.repeats<2:raise ValueError('At least two identical calls are required')
    if not 1<=args.steps<=49:
        raise ValueError('Probe must finish before the first rolling refresh at50')
    source=args.source.resolve(strict=True);run=args.run.resolve(strict=True)
    sys.path[:0]=[str(source),str(source/'src')]
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.losses import make_step,make_loss_vector,LOSS_NAMES
    from examples.particle_corrosion2d.online import FIXED_KEYS,GRAPH_KEYS
    from nonlocal_pinn.training.optimizers import make_optimizer
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    c.train.gpu_policy=args.gpu_policy
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if jax.devices()[0].platform!='gpu':raise ValueError('GPU required')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=list((source/'examples/particle_corrosion2d').glob('*.py'))
    sources.extend(source/name for name in (
        'src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/nn/conditional.py',
        'src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py'))
    for path in sources:
        target=args.output/'source'/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    shutil.copyfile(__file__,args.output/'probe.py')

    rng=json.loads((run/'runtime.json').read_text())['seed']
    net=eqx.tree_deserialise_leaves(run/'model_initial.eqx',build_network(c,jax.random.PRNGKey(rng)))
    opt=make_optimizer(c)
    state=eqx.tree_deserialise_leaves(run/'optimizer_initial.eqx',opt.init(eqx.filter(net,eqx.is_array)))
    weights=jnp.ones(len(LOSS_NAMES),dtype=jnp.float32)
    with np.load(run/'sampling.npz') as f:
        ids,pools,times=(f[k][:args.steps] for k in ('condition_ids','pool_ids','times'))
    with np.load(run/'online_initial_inputs.npz') as f:
        bank={k:f[k] for k in f.files}
    step=make_step(make_loss_vector(c),opt,c,LOSS_NAMES)
    if args.fixed_adjoint:
        from examples.particle_corrosion2d.spatial_adjoint import prepare_adjoint
        prepare=jax.jit(jax.vmap(prepare_adjoint))
    records=[];repeat_records=[]
    for i in range(args.steps):
        data={k:bank[k][ids[i]] for k in FIXED_KEYS}
        data.update({k:bank[k][ids[i],pools[i]] for k in GRAPH_KEYS})
        data['pde_core_points']=np.take_along_axis(data['graph_points'],data['core_indices'][...,None],axis=1)
        payload={k:jnp.asarray(v) for k,v in data.items()}
        if args.fixed_adjoint:
            payload=prepare(payload)
        stamp=time.perf_counter()
        if i==0:
            outputs=[]
            for repeat in range(args.repeats):
                result=step(net,state,weights,jnp.asarray(times[i]),payload,jnp.asarray(True))
                jax.block_until_ready(result)
                arrays=[np.asarray(v) for v in jax.tree.leaves(result) if eqx.is_array(v)]
                if outputs:
                    first=outputs[0]
                    equal=all(np.array_equal(a,b) for a,b in zip(first,arrays,strict=True))
                    maxabs=max(float(np.max(np.abs(a.astype(float)-b.astype(float)))) for a,b in zip(first,arrays,strict=True))
                    category={}
                    for key,old_value,new_value in zip(('model','optimizer','weights','loss','finite'),first_groups,result,strict=True):
                        previous=[np.asarray(v) for v in jax.tree.leaves(old_value) if eqx.is_array(v)]
                        actual=[np.asarray(v) for v in jax.tree.leaves(new_value) if eqx.is_array(v)]
                        category[key]=dict(bitwise_equal=all(np.array_equal(a,b) for a,b in zip(previous,actual,strict=True)),
                            max_abs=max(float(np.max(np.abs(a.astype(float)-b.astype(float)))) for a,b in zip(previous,actual,strict=True)))
                    repeat_records.append(dict(repeat=repeat,bitwise_equal=equal,max_abs_difference=maxabs,categories=category))
                else:
                    outputs.append(arrays);first_groups=result
                if repeat<2:
                    eqx.tree_serialise_leaves(args.output/f'repeat_model_{repeat}.eqx',result[0])
                    eqx.tree_serialise_leaves(args.output/f'repeat_optimizer_{repeat}.eqx',result[1])
                print('REPEAT '+str(repeat),flush=True)
            (args.output/'repeat.json').write_text(json.dumps(repeat_records,indent=2)+'\n')
        net,state,weights,parts,finite=step(net,state,weights,jnp.asarray(times[i]),payload,
                                           jnp.asarray(i%c.train.gradnorm_update_every==0))
        jax.block_until_ready((net,state,weights,parts,finite))
        if not bool(finite):raise FloatingPointError(f'Nonfinite probe update{i+1}')
        row=dict(step=i+1,parts=np.asarray(parts).tolist(),weights=np.asarray(weights).tolist(),
                 seconds=time.perf_counter()-stamp)
        records.append(row)
        if i in (0,args.steps-1):
            eqx.tree_serialise_leaves(args.output/f'model_{i+1:03d}.eqx',net)
            eqx.tree_serialise_leaves(args.output/f'optimizer_{i+1:03d}.eqx',state)
            print(json.dumps(row),flush=True)
    # A second cold trajectory reuses the already-compiled executable, removing
    # source, compilation, initialization and actual sample draws as explanations.
    net=eqx.tree_deserialise_leaves(run/'model_initial.eqx',net)
    state=eqx.tree_deserialise_leaves(run/'optimizer_initial.eqx',state)
    weights=jnp.ones(len(LOSS_NAMES),dtype=jnp.float32)
    replay=[]
    for i in range(args.steps):
        data={k:bank[k][ids[i]] for k in FIXED_KEYS}
        data.update({k:bank[k][ids[i],pools[i]] for k in GRAPH_KEYS})
        data['pde_core_points']=np.take_along_axis(data['graph_points'],data['core_indices'][...,None],axis=1)
        payload={k:jnp.asarray(v) for k,v in data.items()}
        if args.fixed_adjoint:
            payload=prepare(payload)
        net,state,weights,parts,finite=step(net,state,weights,jnp.asarray(times[i]),payload,
                                           jnp.asarray(i%c.train.gradnorm_update_every==0))
        jax.block_until_ready((net,state,weights,parts,finite))
        if not bool(finite):raise FloatingPointError('Nonfinite trajectory replay')
        replay.append(dict(step=i+1,parts=np.asarray(parts).tolist(),weights=np.asarray(weights).tolist()))
    eqx.tree_serialise_leaves(args.output/'trajectory_repeated.eqx',net)
    print('TRAJECTORY repeated',flush=True)
    envkeys=('XLA_FLAGS','JAX_ENABLE_COMPILATION_CACHE','JAX_COMPILATION_CACHE_DIR',
             'XLA_PYTHON_CLIENT_PREALLOCATE','XLA_PYTHON_CLIENT_MEM_FRACTION')
    report=dict(status='passed',fixed_adjoint=args.fixed_adjoint,source=str(source),run=str(run),steps=args.steps,
                environment={k:os.environ[k] if k in os.environ else None for k in envkeys},
                devices=[str(d) for d in jax.devices()],jax_version=jax.__version__,records=records,repeat_records=repeat_records,repeated_trajectory=replay)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
