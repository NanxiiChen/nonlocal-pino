"""Measured exact core+halo NN pilot; sampled fits always check full AT2."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.sparse import load_npz, diags

from examples.fracture2d.asymmetric_damage_range import make_asymmetric_lbfgs
from examples.fracture2d.compact_damage_loading import history_feature_input
from examples.fracture2d.conditioned_damage import make_features
from examples.fracture2d.core_halo_damage import (
    pack_rows, predict_rows, make_sampled_objective, make_cached_objective, validate_probabilities,
    crack_sampling_probabilities,
)
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_residual_weighting import make_weights, build_core_halo, core_matvec
from examples.fracture2d.local_damage_trial import load_geometry, sha
from examples.fracture2d.plain_damage_head import fresh_network, make_problem
from examples.fracture2d.range_damage_optimization import damage_diagnostics, make_augmented_objective
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def prepare_trial(c):
    """Restore and audit fixed-H source; shared by L-BFGS and Adam pilots."""
    a=c.core
    production, source = Path(c.loading.output_dir), Path(a.source_dir)
    identity = json.loads((production/'identity.json').read_text())
    expected = c.to_dict();del expected['core']
    if expected != identity['config']: raise ValueError('physical/source configuration differs')
    for name,digest in identity['code'].items():
        if sha(Path(__file__).with_name(name)) != digest: raise ValueError(f'protected numerical source changed:{name}')
    if sha(source/'current_fields.npz') != a.fields_sha256 or sha(source/'current_network.eqx') != a.network_sha256:
        raise ValueError('core-halo source checkpoint checksum differs')
    audit=json.loads(Path(a.source_audit).read_text())
    if (not audit['pass_replay'] or audit['current_outer']!=a.source_outer
            or audit['source']['fields_sha256']!=a.fields_sha256
            or audit['source']['network_sha256']!=a.network_sha256):
        raise ValueError('core-halo requires independently verified source fields/network')
    mechanics,operator,geometry=load_geometry(c)
    if geometry!=identity['geometry'] or sha(c.cold.basis_path)!=c.cold.basis_sha256:
        raise ValueError('core-halo geometry/basis differs')
    matrix=load_npz(c.cold.basis_path)
    n=mechanics.n_material;points=mechanics.coordinates[:n]
    gc0,gc1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    with np.load(source/'current_fields.npz') as f:
        H,saved_H,seed,saved_image=[f[k].copy() for k in ['H','H_used','d','damage_image']]
    if (H.shape!=(n,) or seed.shape!=(n,) or H.dtype!=np.float64 or seed.dtype!=np.float64
            or not np.isfinite(H).all() or H.min()<0 or H.max()<=0 or not np.isfinite(seed).all()):
        raise ValueError('core-halo requires finite ordered source damage/positive own history')
    with np.load(production/'parameterization.npz') as f:scales,gain=f['column_scale'].copy(),float(f['trunk_gain'])
    features,info=make_features(operator,points,gc0=gc0,length=c.physical.l0,tip=list(c.conditioned.tip),
        powers=list(c.conditioned.powers),roundoff_factor=c.validation.roundoff_factor)
    def known_features(H,load):
        return features(history_feature_input(H,load,c.cold.reference_load,c.cold.history_feature_scaling))
    template=fresh_network(c,info['input_dimension'],matrix,head='identity')
    initial=eqx.tree_deserialise_leaves(source/'current_network.eqx',template)
    s=dict(matrix=matrix,operator=operator,reference_load=c.cold.reference_load,features=known_features)
    original_image,_,objectives,_=make_problem(c,s,scales,gain,head='identity')
    original_predict=eqx.filter_jit(original_image)
    H=jax.lax.stop_gradient(jnp.asarray(H));load=jnp.asarray(a.load,jnp.float64)
    np.testing.assert_allclose(original_predict(initial,jnp.asarray(saved_H),load),saved_image,rtol=1e-10,atol=1e-12)
    X=jax.lax.stop_gradient(known_features(H,load))
    scaled_B=matrix@diags(scales)
    basis_width=int(np.max(np.diff(scaled_B.indptr)))
    bc,bv=pack_rows(scaled_B,width=basis_width);bc,bv=jnp.asarray(bc),jnp.asarray(bv)
    def cached_predict(net):return predict_rows(net,X,bc,bv,trunk_gain=gain)
    cached_predict=eqx.filter_jit(cached_predict)
    np.testing.assert_allclose(cached_predict(initial),original_predict(initial,H,load),rtol=1e-11,atol=1e-12)
    initial_damage=np.asarray(cached_predict(initial))
    matvec,packing=make_padded_matvec(operator);matvec=jax.jit(matvec)
    D=operator.to_scipy();row_width=packing['row_width']+1
    def metrics(net,context,load):
        return damage_diagnostics(cached_predict(net),context['H'],matvec,gc0=gc0,gc1=gc1,
            volumes=jnp.asarray(mechanics.volumes),thresholds=tuple(c.range.activity_thresholds))
    zeros=jnp.zeros(n,jnp.float64);penalty=jnp.asarray(a.penalty,jnp.float64)
    full_context=dict(H=H,lower=zeros,upper=zeros,penalty=penalty)
    full_augmented=make_augmented_objective(original_image,objectives[c.range.method],gc0=gc0)
    def original_objective(net,context,load):
        return full_augmented(net,(context['H'],context['lower'],context['upper'],context['penalty']),load)
    cached_objective=make_cached_objective(cached_predict,matvec,gc0=gc0,gc1=gc1,power=a.power,floor=c.conditioned.loss_floor)
    sampled_objective=make_sampled_objective(gc0=gc0,gc1=gc1,power=a.power,floor=c.conditioned.loss_floor,
        trunk_gain=gain,population_size=n)
    return SimpleNamespace(initial=initial,initial_damage=initial_damage,H=H,load=load,n=n,points=points,seed=seed,D=D,row_width=row_width,X=X,bc=bc,bv=bv,gain=gain,geometry=geometry,mechanics=mechanics,known_features=known_features,matvec=matvec,original_objective=original_objective,cached_objective=cached_objective,sampled_objective=sampled_objective,cached_predict=cached_predict,metrics=metrics,full_context=full_context)


def draw_context(problem, rng, probabilities, *, count, capacity, band_width):
    """Fresh with-replacement core, complete original CSR halo and fixed padding."""
    p=problem
    core=rng.choice(p.n,size=count,replace=True,p=probabilities).astype(np.int32)
    packed=build_core_halo(p.D,core,node_capacity=capacity,row_width=p.row_width)
    ids=packed['global_ids'];size=int(packed['node_mask'].sum())
    context=dict(H=p.H,core_H=p.H[core],features=p.X[ids],basis_columns=p.bc[ids],basis_values=p.bv[ids],
        core_local=jnp.asarray(packed['core_local']),operator_columns=jnp.asarray(packed['columns']),
        operator_values=jnp.asarray(packed['values']),p_core=jnp.asarray(probabilities[core]),
        lower=p.full_context['lower'][core],upper=p.full_context['upper'][core],penalty=p.full_context['penalty'])
    stats=dict(core_count=count,unique_core=int(len(np.unique(core))),actual_halo=size,
        padded_halo=capacity,operator_nonzeros=int(np.count_nonzero(packed['values'])),
        core_fraction_ge_05=float(np.mean(p.seed[core]>=.5)),
        core_fraction_in_band=float(np.mean(abs(p.points[core,1]-.5)<=band_width)))
    return context,core,stats


def run(c):
    a = c.core
    allowed = {'full_original','full_cached','sample_uniform','sample_crack','sample_crack_8','sample_crack_32'}
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform != 'gpu'
            or not a.cases or len(set(a.cases)) != len(a.cases) or not set(a.cases) <= allowed
            or set(a.block_iterations) != set(a.cases)
            or a.sampling_profile not in ('damage_gaussian','geometric_inverse_power')
            or a.load != .0053 or a.power != 16 or not 0 < a.probability_mixture < 1
            or any(type(x) is not int or x < 1 for x in
                   [a.count,a.total_iterations,a.benchmark_evaluations,a.halo_capacity,*a.block_iterations.values()])
            or any(a.total_iterations % x for x in a.block_iterations.values())
            or not np.isfinite(a.penalty) or a.penalty <= 0 or type(a.seed) is not int):
        raise ValueError('core-halo requires explicit CUDA/x64 fixed-H controls and integer resample intervals')
    problem=prepare_trial(c)
    initial=problem.initial
    initial_damage=problem.initial_damage
    H=problem.H
    load=problem.load
    n=problem.n
    points=problem.points
    seed=problem.seed
    D=problem.D
    row_width=problem.row_width
    X=problem.X
    bc=problem.bc
    bv=problem.bv
    gain=problem.gain
    geometry=problem.geometry
    original_objective=problem.original_objective
    cached_objective=problem.cached_objective
    sampled_objective=problem.sampled_objective
    cached_predict=problem.cached_predict
    metrics=problem.metrics
    full_context=problem.full_context
    fitters={}
    for label,obj in [('full_original',original_objective),('full_cached',cached_objective),('sampled',sampled_objective)]:
        fitters[label]=make_asymmetric_lbfgs(initial,obj,metrics)
    flat,_=ravel_pytree(eqx.filter(initial,eqx.is_inexact_array))
    v0,g0=fitters['full_original'][1](flat,full_context,load)
    v1,g1=fitters['full_cached'][1](flat,full_context,load)
    np.testing.assert_allclose(v0,v1,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(g0,g1,rtol=1e-8,atol=1e-9)
    if a.sampling_profile == 'geometric_inverse_power':
        crack_p=np.asarray(crack_sampling_probabilities(jnp.asarray(points),start=a.segment_start,
            end=a.segment_end,width=a.band_width,boost=a.boost,power=a.density_power))
    else:
        static=make_weights(jnp.asarray(points),jnp.asarray(seed),mode='static',boost=a.boost,damage_power=a.damage_power,
            start=a.segment_start,end=a.segment_end,width=a.band_width)
        dynamic=make_weights(jnp.asarray(points),jnp.asarray(seed),mode='damage',boost=a.boost,damage_power=a.damage_power,
            start=a.segment_start,end=a.segment_end,width=a.band_width)
        score=jax.lax.stop_gradient(static*dynamic)
        crack_p=np.asarray(jax.lax.stop_gradient((1-a.probability_mixture)/n+a.probability_mixture*score/jnp.sum(score)))
    probabilities={mode:(np.full(n,1/n,np.float64) if mode=='sample_uniform' else crack_p)
        for mode in a.cases if mode.startswith('sample_')}
    for p in probabilities.values():validate_probabilities(p,population_size=n)
    capacity=a.halo_capacity
    if capacity > n:raise ValueError('padded halo capacity exceeds population')
    def new_context(rng,p):
        return draw_context(problem,rng,p,count=a.count,capacity=capacity,band_width=a.band_width)
    pool_stats={}
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'identity.json',dict(config=c.to_dict(),source_fields_sha256=a.fields_sha256,
        source_network_sha256=a.network_sha256,source_outer=a.source_outer,load=a.load,
        exact_original_support=True,full_domain_acceptance=True,full_history_feature_cache=True,
        detached_probabilities=True,importance_corrected_moment=True,nonlinear_root_unbiased=False,
        optimizer_memory_reset_between_pools=True,fresh_random_draws=True,
        halo_capacity=capacity,core_count=a.count,resampling_in_fit_timing=True,
        physical_state_promoted=False,direct_training_targets=False,geometry=geometry))
    shutil.copyfile(a.source_audit,root/'source_audit.json')
    for name in ['core_halo_damage_trial.py','core_halo_damage.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/name)
    for mode,p in probabilities.items():np.save(root/(mode+'_probabilities.npy'),p)
    print(json.dumps(dict(event='sampling_ready',profile=a.sampling_profile,capacity=capacity)),flush=True)
    reports=[]
    for mode in a.cases:
        sampled=mode.startswith('sample_')
        fit,vg,ev=fitters['sampled'] if sampled else fitters[mode]
        rng=np.random.default_rng(a.seed)
        contexts=[new_context(rng,probabilities[mode])[0]] if sampled else [full_context]
        for context in contexts:
            jax.block_until_ready((vg(flat,context,load),ev(flat,context,load)))
        if sampled:
            context=contexts[0]
            predicted=predict_rows(initial,context['features'],context['basis_columns'],context['basis_values'],trunk_gain=gain)
            # The exact support/NN equality test also covers padding through zero coefficients.
            check_rng=np.random.default_rng(a.seed)
            check_core=check_rng.choice(n,size=a.count,replace=True,p=probabilities[mode])
            np.testing.assert_allclose(core_matvec(predicted,context['operator_columns'],context['operator_values']),
                (D@initial_damage)[check_core],rtol=1e-10,atol=1e-8)
        start=time.perf_counter()
        for _ in range(a.benchmark_evaluations):jax.block_until_ready(vg(flat,contexts[0],load))
        evaluation_seconds=(time.perf_counter()-start)/a.benchmark_evaluations
        net=initial;optimizations=[];core_samples=[];pool_stats[mode]=[]
        rng=np.random.default_rng(a.seed);started=time.perf_counter()
        blocks=a.total_iterations//a.block_iterations[mode] if sampled else 1
        for block in range(1,blocks+1):
            if sampled:
                context,core,stats=new_context(rng,probabilities[mode])
                pool_stats[mode].append(dict(block=block,**stats));core_samples.append(core)
            else:context=full_context
            controls=c.range.lbfgs.to_dict()
            controls['maxiter']=a.block_iterations[mode] if sampled else a.total_iterations
            def progress(row):print(json.dumps(dict(mode=mode,block=block,**row)),flush=True)
            net,opt=fit(net,context,load,controls,progress=progress)
            optimizations.append(dict(block=block,**opt))
            if opt['status']=='residual_reached':break
        fit_seconds=time.perf_counter()-started
        damage=np.asarray(cached_predict(net))
        stats={k:float(v) for k,v in metrics(net,full_context,load).items()}
        ready=(stats['AT2_relative']<=c.range.lbfgs.residual_limit
            and stats['damage_min']>=-c.range.lbfgs.negative_tolerance
            and stats['damage_max']<=1+c.range.lbfgs.positive_tolerance)
        folder=root/mode;folder.mkdir()
        if sampled:np.save(folder/'core_samples.npy',np.asarray(core_samples,np.int32))
        atomic_json(root/'pool_stats.json',pool_stats)
        np.savez_compressed(folder/'fields.npz',d=damage,H=np.asarray(H))
        eqx.tree_serialise_leaves(folder/'network.eqx',net)
        report=dict(mode=mode,ready=ready,metrics=stats,iterations=sum(o['iterations'] for o in optimizations),
            seconds=fit_seconds,value_gradient_seconds=evaluation_seconds,optimizations=optimizations,
            physical_state_promoted=False,sampling_used=sampled,isolated_gpu=True,
            fresh_draw_blocks=len(core_samples),resampling_in_fit_timing=True)
        atomic_json(folder/'result.json',report);reports.append(report)
        atomic_json(root/'result.json',dict(complete=len(reports)==len(a.cases),reports=reports,
            same_H=True,same_initial_network=True,no_loading=True,physical_state_promoted=False))
        print(json.dumps(dict(event='case_complete',mode=mode,iterations=report['iterations'],
            AT2_relative=stats['AT2_relative'],seconds=fit_seconds,value_gradient_seconds=evaluation_seconds,ready=ready)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    args=p.parse_args();jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config']())
