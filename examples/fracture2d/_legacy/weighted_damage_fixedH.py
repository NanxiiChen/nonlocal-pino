"""Same-network/frozen-H uniform and damage-weighted all-node Adam diagnostics."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d._legacy.at2_cg import damage_solver
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config
from examples.fracture2d._legacy.l1_damage_loading import residual_metrics
from examples.fracture2d._legacy.local_damage_trial import sha
from examples.fracture2d._legacy.low_order_damage import make_step, optimizer_count
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.staggered_damage_trial import prepare, read_fields
from examples.fracture2d._legacy.weighted_damage_l1 import frozen_weights, make_weighted_loss


def run(a, config_path):
    t = a.weighting
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or t.physical_state_promoted is not False or t.oracle_evaluation_only is not True
            or t.weight_update != 'frozen_source_physical_damage'
            or type(t.updates) is not int or t.updates < 1 or not 0 < t.region_threshold < 1):
        raise ValueError('weight control requires explicit frozen-H diagnostic protocol')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    p = prepare(a)
    source = Path(t.source_root); repo = Path(__file__).resolve().parents[3]
    for name,digest in t.pins.items():
        if sha(source/name) != digest: raise ValueError(f'weighted source changed: {name}')
    old = json.loads((source/'identity.json').read_text())
    for name,digest in old['code'].items():
        if sha(repo/name) != digest: raise ValueError(f'protected all-node source changed: {name}')
    if old['source_config'] != p['c'].to_dict() or old['geometry'] != p['identity']['geometry']:
        raise ValueError('weighted source model/geometry differs')
    report = json.loads((source/'pinn_full/final/report.json').read_text())
    audit = json.loads((source/'independent_review/audit.json').read_text())
    if (not audit['complete'] or not audit['pass_replay'] or report['load'] != t.load
            or report['optimizer_count'] != t.source_count or not report['q_d_synchronized']):
        raise ValueError('weighted trial needs audited full-domain final state')
    f = read_fields(source/'pinn_full/final/fields.npz')
    net = eqx.tree_deserialise_leaves(source/'pinn_full/final/network.eqx',p['net'])
    state = eqx.tree_deserialise_leaves(source/'pinn_full/final/optimizer.eqx',p['state'])
    if optimizer_count(state,kind='adam') != t.source_count: raise ValueError('weighted Adam count differs')
    m,c,D = p['m'],p['c'],p['D']
    parameterization=read_fields(Path(a.source_root)/'parameterization.npz')
    image,loss=make_weighted_loss(gc0=p['g0'],gc1=p['g1'],gain=float(parameterization['trunk_gain']),population_size=m.n_material)
    image,value=eqx.filter_jit(image),eqx.filter_jit(loss)
    optimizer,schedule=adam_from_config(c.l1_loading);step=make_step(loss,optimizer)
    H=jnp.asarray(f['H']);context=p['full'](H)
    initial=np.asarray(image(net,context))
    np.testing.assert_allclose(initial,f['raw_current_image'],rtol=1e-10,atol=1e-12)
    cases=[ConfigDict(x) for x in t.cases]
    if not cases or len({x.name for x in cases}) != len(cases):raise ValueError('weighted cases must be unique')
    weights={case.name:frozen_weights(f['d'],kind=case.kind,epsilon=case.epsilon,cap=case.cap) for case in cases}
    warm={**context,'weights':jnp.asarray(weights[cases[0].name][0])}
    jax.block_until_ready(step(net,state,warm,jnp.asarray(t.load)))
    root=Path(a.output_dir)
    if (root/'identity.json').exists():raise ValueError('refusing existing weighted diagnostic')
    root.mkdir(parents=True,exist_ok=True)
    code={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):
                code[str(path.relative_to(repo))]=sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))]=sha(config_path)
    for name in code:
        target=root/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
    atomic_json(root/'identity.json',dict(config=a.to_dict(),source_config=c.to_dict(),geometry=old['geometry'],
        code=code,physical_state_promoted=False,history_fixed=True,weights_frozen=True,
        all_node_count=m.n_material,source_count=t.source_count,reference_training_targets=False))
    atomic_json(root/'preflight.json',dict(pass_restore=True,raw_current_image_parity=True,
        optimizer_count=t.source_count,full_nodes=m.n_material))
    solve=damage_solver(p['op'],m.volumes,p['g0'],p['g1'],tol=c.validation.cg_rtol,
        atol=c.validation.cg_atol,maxiter=c.validation.cg_maxiter)
    oracle=np.asarray(solve(H,jnp.asarray(initial)))
    atomic_fields(root/'evaluation_oracle.npz',d=oracle,H=f['H'])
    region=f['d']>=t.region_threshold
    if not region.any():raise ValueError('declared damage evaluation region is empty')
    def metrics(d,w):
        rho=((p['g0']+2*f['H'])*d+p['g1']*(D@d)-2*f['H'])/(2*np.sqrt(np.mean(f['H']**2)))
        delta=d-oracle
        return dict(uniform_L1=float(abs(rho).mean()),weighted_L1=float(w@abs(rho)/w.sum()),
            AT2_relative=residual_metrics(d,f['H'],D,gc0=p['g0'],gc1=p['g1'])[1],
            same_H_damage_L2=float(np.sqrt(m.volumes@delta**2/(m.volumes@oracle**2))),
            region_RMSE=float(np.sqrt(m.volumes[region]@delta[region]**2/m.volumes[region].sum())),
            global_RMSE=float(np.sqrt(m.volumes@delta**2/m.volumes.sum())),damage_min=float(d.min()),damage_max=float(d.max()))
    initial_metrics=metrics(initial,np.ones(m.n_material));summaries=[]
    for case in cases:
        folder=root/case.name;folder.mkdir();w,info=weights[case.name]
        np.save(folder/'weights.npy',w)
        candidate,memory=net,state;ctx={**context,'weights':jnp.asarray(w)};started=time.perf_counter()
        for update in range(1,t.updates+1):
            candidate,memory,objective,gradient=step(candidate,memory,ctx,jnp.asarray(t.load))
            if update in t.checkpoints or update==t.updates:
                d=np.asarray(image(candidate,ctx));m.factors(d)
                if not np.isfinite([float(objective),float(gradient)]).all():raise FloatingPointError('weighted fit is nonfinite')
                count=optimizer_count(memory,kind='adam')
                if count!=t.source_count+update:raise RuntimeError('weighted Adam memory/count reset')
                destination=folder/f'update_{update:05d}';destination.mkdir()
                atomic_fields(destination/'fields.npz',damage_image=d,H=f['H'])
                eqx.tree_serialise_leaves(destination/'network.eqx',candidate)
                eqx.tree_serialise_leaves(destination/'optimizer.eqx',memory)
                row=dict(case=case.name,update=update,load=t.load,**metrics(d,w),weight_statistics=info,
                    weights_sha256=sha(folder/'weights.npy'),optimizer_count=count,learning_rate=float(schedule(count-1)),
                    fields_sha256=sha(destination/'fields.npz'),network_sha256=sha(destination/'network.eqx'),
                    optimizer_sha256=sha(destination/'optimizer.eqx'),seconds=time.perf_counter()-started,
                    physical_state_promoted=False,history_fixed=True,weight_update=t.weight_update)
                atomic_json(destination/'report.json',row)
                atomic_json(root/'result.json',dict(complete=False,current=row,cases=summaries,initial=initial_metrics))
                print(json.dumps(dict(event='weighted_fixed_H',**row)),flush=True)
        summaries.append(row)
    for source,pins in [(Path(a.source_root),a.pins),(Path(t.source_root),t.pins)]:
        for name,digest in pins.items():
            if sha(source/name)!=digest:raise RuntimeError('weighted protected input changed')
    for name,digest in code.items():
        if sha(repo/name)!=digest:raise RuntimeError('weighted numerical source changed')
    atomic_json(root/'result.json',dict(complete=True,cases=summaries,initial=initial_metrics,
        physical_state_promoted=False,load_advanced=False,oracle_evaluation_only=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config'](),args.config)
