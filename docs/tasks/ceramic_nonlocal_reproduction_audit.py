"""CPU-only old/current nonlocal input and symbolic-update comparison."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


def read(path):
    return json.loads(path.read_text())


def run(current, historical, output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.online import FIXED_KEYS, GRAPH_KEYS, build_input
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.losses import make_step, make_loss_vector, LOSS_NAMES
    from nonlocal_pinn.training.optimizers import make_optimizer
    if any(d.platform!='cpu' for d in jax.devices()):
        raise ValueError('This audit must not contend with the active GPU training')
    started=time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    configs=[ConfigDict(read(p/'config.json')) for p in (current,historical)]
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',configs[0].train.matmul_precision)
    runtime=read(current/'runtime.json')
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (current/name).read_bytes()!=(historical/name).read_bytes():
            raise ValueError('Cold state differs: '+name)
    with np.load(current/'sampling.npz') as a,np.load(historical/'sampling.npz') as b:
        for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        ids,pools,times=(a[k][0] for k in ('condition_ids','pool_ids','times'))
    with np.load(current/'online_initial_inputs.npz') as a,np.load(historical/'online_initial_inputs.npz') as b:
        for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        data={k:a[k][ids] for k in FIXED_KEYS}
        data.update({k:a[k][ids,pools] for k in GRAPH_KEYS})
    data['pde_core_points']=np.take_along_axis(data['graph_points'],data['core_indices'][...,None],axis=1)
    payload={k:jnp.asarray(v) for k,v in data.items()}
    net=eqx.tree_deserialise_leaves(current/'model_initial.eqx',build_network(configs[0],jax.random.PRNGKey(42)))
    texts=[]
    for label,c in zip(('current','historical'),configs,strict=True):
        optimizer=make_optimizer(c)
        state=eqx.tree_deserialise_leaves(current/'optimizer_initial.eqx',optimizer.init(eqx.filter(net,eqx.is_array)))
        step=make_step(make_loss_vector(c),optimizer,c,LOSS_NAMES)
        text=step.lower(net,state,jnp.ones(5),jnp.asarray(times),payload,jnp.asarray(True)).as_text()
        (output/(label+'-update.mlir')).write_text(text)
        texts.append(text)
    if texts[0]!=texts[1]:
        raise ValueError('Configured CPU-lowered update graphs differ; inspect saved MLIR')
    # Load the frozen old constructor under an isolated package name. Shared
    # nonlocal_pinn modules are byte-checked before this optional-path comparison.
    shared=('src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/nn/conditional.py',
            'src/nonlocal_pinn/training/optimizers.py','examples/particle_corrosion2d/losses.py',
            'examples/particle_corrosion2d/network.py','examples/particle_corrosion2d/strategies.py',
            'examples/particle_corrosion2d/operators.py','examples/particle_corrosion2d/dimensional.py')
    for name in shared:
        if (current/'source'/name).read_bytes()!=(historical/'source'/name).read_bytes():
            raise ValueError('Shared update source differs: '+name)
    package=types.ModuleType('frozen_particle_reproduction')
    package.__path__=[str(historical/'source/examples/particle_corrosion2d')]
    sys.modules[package.__name__]=package
    prior_builder=importlib.import_module(package.__name__+'.online').build_input
    events=[json.loads(line) for line in (current/'online_refresh.jsonl').read_text().splitlines()]
    chosen=[events[i] for i in sorted(set((0,1,4,len(events)//2,len(events)-1)))]
    manifest=read(Path(runtime['inputs'])/'manifest.json')
    checked=[]
    for event in chosen:
        record=event['records'][0]
        draw,revision=record['draw_id'],record['spatial_version']
        geometry=None if draw>=0 else manifest['geometries'][-1-draw]
        new,new_meta=build_input(configs[0].to_dict(),runtime['seed'],draw,revision,geometry)
        old,old_meta=prior_builder(configs[1].to_dict(),runtime['seed'],draw,revision,geometry)
        if new_meta['geometry']!=old_meta['geometry']:raise ValueError('Rolling geometry changed')
        if set(new)!=set(old):raise ValueError('Rolling payload schema changed')
        for name in new:np.testing.assert_array_equal(new[name],old[name],err_msg=name)
        checked.append(dict(draw_id=draw,spatial_version=revision,completed_updates=event['completed_updates']))
    report=dict(status='passed',current=str(current),historical=str(historical),
        identical_initial_state=True,identical_initial_bank=True,identical_all_planned_time_slot_draws=True,
        identical_symbolic_update_graph=True,lowered_graph_characters=len(texts[0]),
        checked_rolling_payloads=checked,shared_source_bytes=list(shared),
        limitation='CPU lowering/input equivalence does not guarantee cross-GPU FP32 execution identity or prove its causal effect on convergence.',
        seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('current','historical','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.current,a.historical,a.output)
