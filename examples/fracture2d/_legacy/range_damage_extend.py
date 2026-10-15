"""Continue certified same-H linear network and inequality AL duals."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz

from examples.fracture2d._legacy.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import load_geometry,sha
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network,make_problem
from examples.fracture2d._legacy.range_damage_optimization import (
    damage_diagnostics,make_augmented_objective,update_duals,
)
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def run(c):
    if (jax.devices()[0].platform!='gpu' or not jax.config.jax_enable_x64
            or c.range.method!='residual_l16' or c.head.load!=.0001
            or list(c.extension.penalties)!=[1e4,1e5,1e6]
            or c.range.lbfgs.maxiter!=2048 or c.range.lbfgs.residual_limit!=.001
            or c.range.lbfgs.range_tolerance!=1e-10
            or len(c.range.activity_thresholds)==0
            or any(not np.isfinite(x) or not 0<x<1 for x in c.range.activity_thresholds)):
        raise ValueError('range trial requires declared same-H controls and original gates')
    source=Path(c.extension.source_dir)
    required={'identity.json','audit.json','own_fields.npz','parameterization.npz',
        'AL_0004/network.eqx','AL_0004/fields.npz','result.json'}
    if set(c.extension.source)!=required:
        raise ValueError('range trial requires all explicit certified neural source pins')
    protected={}
    for name,digest in c.extension.source.items():
        path=source/name
        if sha(path)!=digest:
            raise ValueError(f'range trial source changed: {path}')
        protected[str(path)]=digest
    original=json.loads((source/'identity.json').read_text())
    audit=json.loads((source/'audit.json').read_text())
    if (not audit['pass_replay'] or not audit['exact_same_own_H']
            or not original['no_direct_targets'] or not original['no_output_projection']
            or original['physical_state_promoted']):
        raise ValueError('range trial source must be audited same-own-H neural diagnostic')
    for key in ('physical','geometry','nl','trial','conditioned','validation','arch','train','head','cold','range'):
        if original['config'][key]!=c[key].to_dict():
            raise ValueError(f'range trial source/config mismatch: {key}')
    mechanics,operator,geometry=load_geometry(c)
    with np.load(source/'own_fields.npz',allow_pickle=False) as f:
        q,d_physical,H=[f[key].copy() for key in ('q','d_physical','H_used')]
    if (H.shape!=(mechanics.n_material,) or q.shape!=mechanics.load_direction.shape
            or not np.isfinite(H).all() or H.min()<0 or H.max()<=0):
        raise ValueError('range trial requires ordered finite positive own history')
    np.testing.assert_array_equal(d_physical,np.zeros(mechanics.n_material))
    np.testing.assert_array_equal(H,mechanics.driver(q))
    basis=Path(c.cold.basis_path)
    if sha(basis)!=c.cold.basis_sha256:
        raise ValueError('range trial compact geometry changed')
    protected[str(basis)]=sha(basis)
    for name in ('identity.json','mesh.npz','damage_operator.npz','fenics_parity.json'):
        path=Path(c.trial.geometry_dir)/name
        protected[str(path)]=sha(path)
    matrix=load_npz(basis)
    gc0,gc1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    features,info=make_features(operator,mechanics.coordinates[:mechanics.n_material],
        gc0=gc0,length=c.physical.l0,tip=list(c.conditioned.tip),powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    with np.load(source/'parameterization.npz',allow_pickle=False) as f:
        columns,gain=f['column_scale'].copy(),float(f['trunk_gain'])
    s=dict(matrix=matrix,operator=operator,features=lambda known,load:features(known),reference_load=c.head.load)
    template=fresh_network(c,info['input_dimension'],matrix,head='identity')
    initial=eqx.tree_deserialise_leaves(source/'AL_0004/network.eqx',template)
    image,base_metrics,objectives,_=make_problem(c,s,columns,gain,head='identity')
    predict=eqx.filter_jit(image)
    matvec,_=make_padded_matvec(operator)
    matvec=jax.jit(matvec)
    with np.load(Path(c.trial.geometry_dir)/'mesh.npz',allow_pickle=False) as f:
        volumes=jnp.asarray(f['volumes'])
    if (volumes.shape!=H.shape or not np.isfinite(np.asarray(volumes)).all()
            or float(volumes.min())<=0):
        raise ValueError('range activity diagnostics require positive matching material volumes')
    thresholds=tuple(c.range.activity_thresholds)
    def field_metrics(damage,history):
        return damage_diagnostics(damage,history,matvec,gc0=gc0,gc1=gc1,
            volumes=volumes,thresholds=thresholds)
    def metrics(net,history,load):
        return field_metrics(image(net,history,load),history)
    measure,measure_field=eqx.filter_jit(metrics),jax.jit(field_metrics)
    H,load=jnp.asarray(H),jnp.asarray(c.head.load,jnp.float64)
    with np.load(source/'AL_0004/fields.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['H_used'],H)
        np.testing.assert_allclose(predict(initial,H,load),f['damage'],rtol=1e-10,atol=1e-12)
        initial_lower,initial_upper=[jnp.asarray(f[k]) for k in ('lower_after','upper_after')]
    if (initial_lower.shape!=H.shape or initial_upper.shape!=H.shape
            or not np.isfinite(np.asarray(initial_lower)).all()
            or not np.isfinite(np.asarray(initial_upper)).all()
            or float(initial_lower.min())<0 or float(initial_upper.min())<0):
        raise ValueError('AL continuation requires finite nonnegative matching duals')
    previous=json.loads((source/'result.json').read_text())
    if not previous['complete'] or previous['fixed_H_ready'] or previous['physical_state_promoted']:
        raise ValueError('AL extension requires completed unready diagnostic parent')
    root=Path(c.extension.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    names=['range_damage_extend.py','range_damage_trial.py','range_damage_optimization.py','plain_damage_head.py',
        'plain_sigmoid_damage.py','cached_damage_optimization.py','conditioned_damage.py',
        'compact_damage_features.py','compact_damage_loading.py','local_damage_trial.py',
        'local_mechanics.py','output_damage_continue.py','split_pinn_gradients.py','train.py']
    code={name:sha(Path(__file__).with_name(name)) for name in names}
    atomic_json(root/'identity.json',dict(schema='linear-damage-range-AL-extend-v1',config=c.to_dict(),
        source=protected,code=code,geometry=geometry,features=info,same_own_H=True,
        trainable_parameters=sum(a.size for a in jax.tree.leaves(eqx.filter(initial,eqx.is_inexact_array))),
        no_direct_targets=True,no_output_projection=True,no_damage_correction=True,
        physical_state_promoted=False))
    (root/'code_snapshot').mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    atomic_fields(root/'own_fields.npz',q=q,d_physical=d_physical,H_used=np.asarray(H))
    np.savez_compressed(root/'parameterization.npz',column_scale=columns,trunk_gain=gain)
    eqx.tree_serialise_leaves(root/'initial_network.eqx',initial)
    def numbers(values):
        return {key:float(value) for key,value in values.items()}
    comparisons=dict(zero_damage=numbers(measure_field(jnp.zeros_like(H),H)),
        all_one_damage=numbers(measure_field(jnp.ones_like(H),H)),
        initial_identity=numbers(measure(initial,H,load)))
    atomic_json(root/'activity_comparison.json',comparisons)
    atomic_fields(root/'initial_duals.npz',lower=np.asarray(initial_lower),upper=np.asarray(initial_upper))
    def augmented_metrics(net,context,load):
        return metrics(net,context[0],load)
    augmented=make_augmented_objective(image,objectives[c.range.method],gc0=gc0)
    fit_augmented,_,_=make_cached_lbfgs(initial,augmented,augmented_metrics)
    net=initial
    lower,upper=initial_lower,initial_upper
    reports=[]
    for stage,penalty_value in enumerate(c.extension.penalties,start=1):
        folder=root/f'AL_{stage:04d}'
        folder.mkdir()
        penalty=jnp.asarray(penalty_value,jnp.float64)
        context=(H,lower,upper,penalty)
        atomic_fields(folder/'initial_context.npz',H_used=np.asarray(H),lower=np.asarray(lower),
            upper=np.asarray(upper),penalty=np.asarray(penalty))
        started=time.perf_counter()
        def progress(row):
            print(json.dumps(dict(event='augmented',stage=stage,penalty=penalty_value,**row)),flush=True)
        net,optimization=fit_augmented(net,context,load,c.range.lbfgs.to_dict(),progress=progress)
        damage=predict(net,H,load)
        lower,upper=update_duals(damage,H,lower,upper,penalty,gc0=gc0)
        atomic_fields(folder/'fields.npz',damage=np.asarray(damage),H_used=np.asarray(H),
            lower_after=np.asarray(lower),upper_after=np.asarray(upper))
        eqx.tree_serialise_leaves(folder/'network.eqx',net)
        ready=optimization['status']=='residual_reached'
        report=dict(stage=stage,penalty=penalty_value,status=optimization['status'],ready=ready,
            metrics=numbers(measure(net,H,load)),optimization=optimization,
            physical_state_promoted=False,seconds=time.perf_counter()-started,
            fields_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'))
        atomic_json(folder/'result.json',report)
        reports.append(report)
        atomic_json(root/'result.json',dict(complete=False,reports=reports,
            fixed_H_ready=ready,physical_state_promoted=False))
        print(json.dumps(dict(event='stage_complete',stage=stage,ready=ready,**report['metrics'])),flush=True)
        if ready:
            break
    if (protected!={path:sha(path) for path in protected}
            or code!={name:sha(Path(__file__).with_name(name)) for name in names}):
        raise RuntimeError('range trial source/code changed')
    atomic_json(root/'result.json',dict(complete=True,reports=reports,
        fixed_H_ready=reports[-1]['ready'],physical_state_promoted=False))
    print(json.dumps(dict(event='complete',fixed_H_ready=reports[-1]['ready'])),flush=True)
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    c=runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    run(c)
