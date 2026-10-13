"""Fixed-H persistent Adam, sampled/full controls and original-gate polishing."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from examples.fracture2d.asymmetric_damage_range import make_asymmetric_lbfgs
from examples.fracture2d.core_halo_adam import make_adam
from examples.fracture2d.core_halo_damage import crack_sampling_probabilities, validate_probabilities
from examples.fracture2d.core_halo_damage_trial import prepare_trial, draw_context
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_residual_weighting import core_matvec
from examples.fracture2d.local_damage_trial import sha


def run(c):
    a=c.core;s=a.adam
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform!='gpu'
            or a.sampling_profile!='geometric_inverse_power' or a.load!=.0053 or a.power!=16
            or any(type(x) is not int or x<1 for x in
                [s.total_iterations,s.resample_every,s.check_every,s.polish_iterations,a.count,a.halo_capacity])
            or not s.rates or len(set(s.rates))!=len(s.rates)
            or not np.isfinite(s.rates).all() or min(s.rates)<=0
            or not np.isfinite(s.minimum_fraction) or not 0<s.minimum_fraction<=1):
        raise ValueError('Adam trial requires explicit isolated CUDA/x64 controls and positive rates/budgets')
    p=prepare_trial(c)
    if a.halo_capacity>p.n:raise ValueError('padded halo exceeds population')
    probabilities=np.asarray(crack_sampling_probabilities(jnp.asarray(p.points),start=a.segment_start,
        end=a.segment_end,width=a.band_width,boost=a.boost,power=a.density_power))
    validate_probabilities(probabilities,population_size=p.n)
    baseline=Path(s.full_lbfgs_baseline)
    baseline_report=json.loads((baseline/'result.json').read_text())
    baseline_audit=json.loads(Path(s.full_lbfgs_audit).read_text())
    if (not baseline_audit['pass_replay'] or not baseline_audit['same_H']
            or baseline_audit['same_load']!=a.load
            or sha(baseline/'network.eqx')!=s.full_lbfgs_network_sha256
            or baseline_report['mode']!='full_cached'):
        raise ValueError('cached L-BFGS comparison baseline provenance differs')
    with np.load(baseline/'fields.npz') as f:np.testing.assert_array_equal(f['H'],np.asarray(p.H))
    # Independently compare the cached and original full objective/parameter gradient.
    original_vg=eqx.filter_jit(eqx.filter_value_and_grad(p.original_objective))
    cached_vg=eqx.filter_jit(eqx.filter_value_and_grad(p.cached_objective))
    v0,g0=original_vg(p.initial,p.full_context,p.load)
    v1,g1=cached_vg(p.initial,p.full_context,p.load)
    np.testing.assert_allclose(v0,v1,rtol=1e-10,atol=1e-12)
    for x,y in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)):
        np.testing.assert_allclose(x,y,rtol=1e-8,atol=1e-9)
    metrics=eqx.filter_jit(p.metrics)
    def read_metrics(net):
        result={k:float(v) for k,v in metrics(net,p.full_context,p.load).items()}
        if not np.isfinite(list(result.values())).all():raise FloatingPointError('nonfinite full Adam metrics')
        return result
    def range_pass(values):
        return (values['damage_min']>=-c.range.lbfgs.negative_tolerance
            and values['damage_max']<=1+c.range.lbfgs.positive_tolerance)
    def ready(values):return range_pass(values) and values['AT2_relative']<=c.range.lbfgs.residual_limit
    def score(values):return (not range_pass(values),values['AT2_relative'])
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'identity.json',dict(config=c.to_dict(),source_fields_sha256=a.fields_sha256,
        source_network_sha256=a.network_sha256,source_outer=a.source_outer,load=a.load,
        exact_original_support=True,full_domain_acceptance=True,full_history_feature_cache=True,
        detached_probabilities=True,importance_corrected_moment=True,nonlinear_root_unbiased=False,
        optimizer_memory_reset_between_pools=False,adam_state_persistent=True,fresh_random_draws=True,
        resampling_in_fit_timing=True,physical_state_promoted=False,direct_training_targets=False,geometry=p.geometry))
    np.save(root/'probabilities.npy',probabilities)
    shutil.copyfile(a.source_audit,root/'source_audit.json')
    for name in ['core_halo_adam_trial.py','core_halo_adam.py','core_halo_damage_trial.py','core_halo_damage.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/name)
    reports=[];sampled_candidates=[]

    def fit_adam(mode,lr,*,sampled):
        optimizer,step,schedule=make_adam(p.sampled_objective if sampled else p.cached_objective,
            learning_rate=lr,decay_steps=s.decay_steps,decay_rate=s.decay_rate,
            minimum_rate=lr*s.minimum_fraction,b1=s.b1,b2=s.b2,eps=s.eps,eps_root=s.eps_root,grad_clip=s.grad_clip)
        net=p.initial;state=optimizer.init(eqx.filter(net,eqx.is_inexact_array))
        rng=np.random.default_rng(a.seed)
        context,first_core,_=draw_context(p,rng,probabilities,count=a.count,capacity=a.halo_capacity,
            band_width=a.band_width) if sampled else (p.full_context,None,None)
        if sampled:
            from examples.fracture2d.core_halo_damage import predict_rows
            halo=predict_rows(net,context['features'],context['basis_columns'],context['basis_values'],trunk_gain=p.gain)
            np.testing.assert_allclose(core_matvec(halo,context['operator_columns'],context['operator_values']),
                (p.D@np.asarray(p.cached_predict(net)))[first_core],rtol=1e-10,atol=1e-8)
        jax.block_until_ready(step(net,state,context,p.load))
        current_metrics=read_metrics(net);best_net,best_state,best_metrics,best_iteration=net,state,current_metrics,0
        trace=[dict(iteration=0,seconds=0.,ready=ready(current_metrics),adam_count=0,**current_metrics)]
        cores=[];pool_stats=[];rng=np.random.default_rng(a.seed);started=time.perf_counter();iteration=0
        for iteration in range(1,s.total_iterations+1):
            if sampled and (iteration-1)%s.resample_every==0:
                context,core,stats=draw_context(p,rng,probabilities,count=a.count,capacity=a.halo_capacity,
                    band_width=a.band_width)
                cores.append(core);pool_stats.append(dict(first_iteration=iteration,**stats))
            net,state,value,gradient_norm=step(net,state,context,p.load)
            if not np.isfinite([float(value),float(gradient_norm)]).all():
                raise FloatingPointError(f'nonfinite Adam loss/gradient:{mode}/iteration{iteration}')
            if iteration%s.check_every==0 or iteration==s.total_iterations:
                current_metrics=read_metrics(net)
                count,schedule_count=int(state[1][0].count),int(state[1][1].count)
                if count!=iteration or schedule_count!=iteration:raise RuntimeError('Adam state reset/count drift')
                if score(current_metrics)<score(best_metrics):
                    best_net,best_state,best_metrics,best_iteration=net,state,current_metrics,iteration
                row=dict(iteration=iteration,seconds=time.perf_counter()-started,ready=ready(current_metrics),
                    adam_count=count,learning_rate=float(schedule(iteration-1)),sampled_loss=float(value),
                    gradient_norm=float(gradient_norm),**current_metrics)
                trace.append(row);print(json.dumps(dict(mode=mode,**row)),flush=True)
                if ready(current_metrics):break
        seconds=time.perf_counter()-started
        final_metrics=read_metrics(net);folder=root/mode;folder.mkdir()
        for label,m,st in [('final',net,state),('best',best_net,best_state)]:
            eqx.tree_serialise_leaves(folder/(label+'_network.eqx'),m)
            eqx.tree_serialise_leaves(folder/(label+'_adam_state.eqx'),st)
            np.savez_compressed(folder/(label+'_fields.npz'),d=np.asarray(p.cached_predict(m)),H=np.asarray(p.H))
        if sampled:np.save(folder/'core_samples.npy',np.asarray(cores,np.int32))
        atomic_json(folder/'pool_stats.json',pool_stats)
        report=dict(mode=mode,learning_rate=lr,iterations=iteration,seconds=seconds,sampling_used=sampled,
            metrics=final_metrics,best_metrics=best_metrics,best_iteration=best_iteration,
            ready=ready(final_metrics),best_ready=ready(best_metrics),trace=trace,fresh_draw_blocks=len(cores),
            adam_count=int(state[1][0].count),schedule_count=int(state[1][1].count),
            adam_state_persistent=True,physical_state_promoted=False,isolated_gpu=True)
        atomic_json(folder/'result.json',report);reports.append(report)
        atomic_json(root/'result.json',dict(complete=False,reports=reports,physical_state_promoted=False))
        print(json.dumps(dict(event='adam_complete',mode=mode,iterations=iteration,seconds=seconds,
            AT2_relative=final_metrics['AT2_relative'],best_AT2=best_metrics['AT2_relative'],best_iteration=best_iteration)),flush=True)
        return best_net,report

    for index,lr in enumerate(s.rates):
        sampled_candidates.append(fit_adam(f'sample_rate_{index}',lr,sampled=True))
    best_sampled,sampled_report=min(sampled_candidates,key=lambda pair:score(pair[1]['best_metrics']))
    selected_rate=sampled_report['learning_rate']
    best_full,full_report=fit_adam('full_adam',selected_rate,sampled=False)
    baseline_net=eqx.tree_deserialise_leaves(baseline/'network.eqx',p.initial)
    np.testing.assert_allclose(p.cached_predict(baseline_net),np.load(baseline/'fields.npz')['d'],rtol=1e-10,atol=1e-12)
    fit,vg,ev=make_asymmetric_lbfgs(p.initial,p.cached_objective,p.metrics)
    flat,_=ravel_pytree(eqx.filter(p.initial,eqx.is_inexact_array))
    jax.block_until_ready((vg(flat,p.full_context,p.load),ev(flat,p.full_context,p.load)))
    polishes=[]
    for mode,net,prior in [('sampled_polish',best_sampled,sampled_report['seconds']),
                         ('full_adam_polish',best_full,full_report['seconds']),
                         ('full_lbfgs_polish',baseline_net,baseline_report['seconds'])]:
        controls=c.range.lbfgs.to_dict();controls['maxiter']=s.polish_iterations
        started=time.perf_counter()
        net,optimization=fit(net,p.full_context,p.load,controls,
            progress=lambda row:print(json.dumps(dict(mode=mode,**row)),flush=True))
        seconds=time.perf_counter()-started;values=read_metrics(net)
        folder=root/mode;folder.mkdir()
        eqx.tree_serialise_leaves(folder/'network.eqx',net)
        np.savez_compressed(folder/'fields.npz',d=np.asarray(p.cached_predict(net)),H=np.asarray(p.H))
        report=dict(mode=mode,metrics=values,ready=ready(values),iterations=optimization['iterations'],
            seconds=seconds,prior_seconds=prior,route_seconds=prior+seconds,optimization=optimization,
            physical_state_promoted=False,direct_training_targets=False,isolated_gpu=True)
        polishes.append(report);atomic_json(folder/'result.json',report)
        print(json.dumps(dict(event='polish_complete',mode=mode,ready=report['ready'],
            AT2_relative=values['AT2_relative'],route_seconds=report['route_seconds'])),flush=True)
    atomic_json(root/'result.json',dict(complete=True,reports=reports,polishes=polishes,
        selected_sample_mode=sampled_report['mode'],selected_rate=selected_rate,same_H=True,same_initial_network=True,
        no_loading=True,physical_state_promoted=False,rate_search_seconds=sum(r['seconds'] for r in reports if r['sampling_used'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config']())
