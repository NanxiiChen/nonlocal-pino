"""Independent saved-checkpoint replay and physical diagnostics, without FEM."""
from pathlib import Path
import argparse
import json
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network,encode,fields,field_jets
from examples.channel_nsch2d.problem import walls,flow_rate


def validate(run):
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    summary=json.loads((run/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('A complete run is required')
    with np.load(run/'prediction.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
    with np.load(run/'sampling.npz',allow_pickle=False) as f:draws={k:f[k] for k in f.files}
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:known={k:f[k] for k in f.files}
    expected={'points','volumes','neighbors','laplacian','interior','amplitudes',
              'boundary','boundary_weights','domain','domain_weights'}
    from examples.channel_nsch2d.problem import boundary_payload_shapes
    expected.update(boundary_payload_shapes(c))
    if 'phase_transport' in c:
        from examples.channel_nsch2d.transport import transport_shapes
        expected.update(transport_shapes(c))
    if 'derivatives' in c:
        from examples.channel_nsch2d.operators import gradient_shapes
        expected.update(gradient_shapes(c))
    if c.nl.method == 'fem_weak_p2p1':
        from examples.channel_nsch2d.fem_weak import shapes
        expected = set(shapes(c))
    if set(known)!=expected:raise ValueError('Unexpected geometry-bank payload')
    online_audit=None
    refinement_audit=None
    initialization_audit=None
    if 'initialization'in c.train:
        from examples.channel_nsch2d.initialization import audit_model_initialization
        initialization_audit=audit_model_initialization(run,c)
        (run/'initialization-validation.json').write_text(json.dumps(initialization_audit,indent=2,allow_nan=False)+'\n')
    if 'refinement' in c.train:
        from examples.channel_nsch2d.refinement import audit_refinement
        refinement_audit=audit_refinement(run,c)
        (run/'refinement-validation.json').write_text(json.dumps(refinement_audit,indent=2,allow_nan=False)+'\n')
    if 'online' in c.operator:
        from examples.channel_nsch2d.online import audit_training_inputs
        online_audit=audit_training_inputs(run,c)
        (run/'online-validation.json').write_text(json.dumps(online_audit,indent=2,allow_nan=False)+'\n')
    if 'causal' in c:
        from examples.channel_nsch2d.causal import audit_trace
        causal_audit=audit_trace(run,c)
        (run/'causal-validation.json').write_text(json.dumps(causal_audit,indent=2,allow_nan=False)+'\n')
    indices=draws['indices'];times=draws['times'];n=c.operator.train_conditions
    start_step=summary['start_step'] if 'start_step' in summary else 0
    if isinstance(start_step,bool) or not isinstance(start_step,int) or not 0<=start_step<c.train.iterations:
        raise ValueError('Invalid global sampling start step')
    if 'continuation' in c.train:
        imported=json.loads((run/'continuation.json').read_text())
        if imported['start_step']!=start_step:raise ValueError('Continuation global step disagrees')
        parent=Path(c.train.continuation.parent_run)
        for initial,final in (('model_initial.eqx','model_final.eqx'),('optimizer_initial.eqx','optimizer_final.eqx')):
            if (run/initial).read_bytes()!=(parent/final).read_bytes():
                raise ValueError(f'Actual imported {initial} bytes disagree with parent state')
    updates=c.train.iterations-start_step
    if indices.shape!=(updates,c.operator.condition_batch) or np.any(indices<0) or np.any(indices>=n):
        raise ValueError('Invalid actual condition draws')
    if times.shape!=(updates,c.sample.Nt):raise ValueError('Invalid actual time draws')
    strata=np.arange(c.sample.Nt)*c.physical.terminal_time/c.sample.Nt
    if 'time_discretization' in c:
        from examples.channel_nsch2d.temporal import step_count
        intervals=draws['intervals'];per=step_count(c)//c.sample.Nt
        if intervals.shape!=times.shape or not np.issubdtype(intervals.dtype,np.integer):
            raise ValueError('Actual discrete interval indices must align with times')
        if not np.all((intervals>=np.arange(c.sample.Nt)*per)&(intervals<(np.arange(c.sample.Nt)+1)*per)):
            raise ValueError('Actual intervals violate integer strata')
        expected=(intervals+1).astype(times.dtype)*np.asarray(c.time_discretization.dt,dtype=times.dtype)
        np.testing.assert_array_equal(times,expected)
    else:
        if not np.all((times>=strata)&(times<strata+c.physical.terminal_time/c.sample.Nt)):
            raise ValueError('Actual times violate declared uniform strata')
    _,data_key=jax.random.split(jax.random.PRNGKey(c.train.seed))
    def actual_keys(step):
        kg,kt=jax.random.split(jax.random.fold_in(data_key,step))
        selected=jax.random.randint(kg,(c.operator.condition_batch,),0,n)
        if 'time_discretization' in c:
            from examples.channel_nsch2d.temporal import sample_intervals
            interval,t=sample_intervals(kt,c)
            return selected,t,interval
        if 'time_derivative' in c:
            from examples.channel_nsch2d.temporal import sample_continuous
            return selected,sample_continuous(kt,c)
        draws=jax.random.uniform(kt,(c.sample.Nt,),dtype=jnp.float32)
        t=(jnp.arange(c.sample.Nt)+draws)*c.physical.terminal_time/c.sample.Nt
        return selected,t
    replay_draws=jax.jit(jax.vmap(actual_keys))(jnp.arange(start_step,c.train.iterations,dtype=jnp.uint32))
    np.testing.assert_array_equal(indices,np.asarray(replay_draws[0]))
    np.testing.assert_array_equal(times,np.asarray(replay_draws[1],dtype=times.dtype))
    if 'time_discretization' in c:np.testing.assert_array_equal(draws['intervals'],np.asarray(replay_draws[2]))
    dtype=np.dtype(c.train.compute_dtype)
    def device(d):
        return {k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else dtype) for k,v in d.items()}
    data,_=geometry_data(saved['amplitudes'],c);data=device(data)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    projected='chemical_constraint'in c
    if not projected:predict=eqx.filter_jit(lambda net,d,x,t:fields(net,encode(net,d),x,t,d['amplitudes'],c))
    point=jnp.asarray(saved['points'],dtype=dtype);time=jnp.asarray(saved['time'],dtype=dtype)
    probe=point[::max(1,len(point)//24)][:24]
    probe_time=jnp.asarray(np.array([1/12,1/2,1.])*c.physical.terminal_time,dtype=dtype)
    roots,weight=np.polynomial.legendre.leggauss(64);eta=(roots+1)/2
    section=np.linspace(0,c.geometry.length,17)
    xx,ee=np.meshgrid(section,eta,indexing='ij');bottom,top=walls(xx,saved['amplitudes'],c,xp=np)
    flow_points=np.stack((xx,bottom+ee*(top-bottom)),axis=-1).reshape(-1,2)
    # Queries are independent (no query self-attention). Concatenating these
    # sets avoids five separate primal compilations, while also checking that
    # replay is invariant to the original saved-query batch size.
    sets=(point,probe,jnp.asarray(flow_points,dtype=dtype),data['boundary'][:,:2],data['points'])
    sizes=[len(a) for a in sets];end=np.cumsum(sizes);start=np.r_[0,end[:-1]]
    slices=[slice(a,b) for a,b in zip(start,end,strict=True)]
    joined=jnp.concatenate(sets,axis=0)
    if projected:
        from examples.channel_nsch2d.projected_fields import make_query_fn,query_rule,make_decoder_fn
        physical_query=make_query_fn(c);decoder=make_decoder_fn(c)
        host={k:np.asarray(v)for k,v in data.items()}
        rule={k:jnp.asarray(v)for k,v in query_rule(host,np.asarray(joined),c).items()}
        predict=eqx.filter_jit(lambda net,d,x,t,r:physical_query(net,encode(net,d),d,x,t,r))
        combined=np.asarray(predict(model,data,joined,time,rule),dtype=np.float64)
        probe_rule={k:jnp.asarray(v)for k,v in query_rule(host,np.asarray(probe),c).items()}
        with np.load(run/'evaluation_query_rule.npz',allow_pickle=False)as f:
            expected_rule=query_rule(host,np.asarray(point),c)
            if set(f.files)!=set(expected_rule):raise ValueError('Saved physical chemical query rule schema differs')
            for k,v in expected_rule.items():np.testing.assert_array_equal(f[k],v)
    else:combined=np.asarray(predict(model,data,joined,time),dtype=np.float64)
    if combined.shape!=(len(time),sum(sizes),5) or not np.isfinite(combined).all():
        raise ValueError('Combined physical replay has invalid shape or nonfinite values')
    replay=combined[:,slices[0]]
    probe_indices=np.argmin(np.abs(saved['time'][:,None]-np.asarray(probe_time)[None,:]),axis=0)
    np.testing.assert_allclose(saved['time'][probe_indices],np.asarray(probe_time),atol=2e-7,rtol=0)
    gap=float(np.max(np.abs(replay-saved['fields'])))
    if not np.isfinite(replay).all() or gap>2e-5:raise ValueError(f'Saved physical replay failed: {gap}')
    # JVP of time must preserve the primal; use an independent small query set.
    if projected:
        @eqx.filter_jit
        def jets(net,d):
            e=encode(net,d)
            value,rate=jax.jvp(lambda ts:physical_query(net,e,d,probe,ts,probe_rule),
                (probe_time,),(jnp.ones_like(probe_time),))
            velocity=lambda x:decoder(net,e,x,probe_time,d['amplitudes'])[...,2:4]
            ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=probe.dtype),probe.shape)
            ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=probe.dtype),probe.shape)
            _,dx=jax.jvp(velocity,(probe,),(ex,));_,dy=jax.jvp(velocity,(probe,),(ey,))
            return value,rate,dx,dy
    else:jets=eqx.filter_jit(lambda net,d:field_jets(net,encode(net,d),probe,probe_time,d['amplitudes'],c))
    evaluated=[np.asarray(v) for v in jets(model,data)]
    if any(not np.isfinite(v).all() for v in evaluated):raise ValueError('Nonfinite independent field/time/spatial derivatives')
    value,_,dx,dy=evaluated
    primal_gap=float(np.max(np.abs(value-combined[probe_indices,slices[1]])))
    if primal_gap>2e-5:raise ValueError(f'Primal/JVP inconsistency: {primal_gap}')
    flow=combined[probe_indices,slices[2]]
    flux=np.sum(flow[...,2].reshape(3,len(section),64)*(top-bottom)[None]*weight/2,axis=-1)
    target=np.broadcast_to(np.asarray(flow_rate(np.asarray(probe_time),c,xp=np))[:,None],flux.shape)
    flux_gap=float(np.max(np.abs(flux-target)))
    boundary=np.asarray(data['boundary']);bc=combined[:,slices[3]]
    wall_gap=float(np.max(np.abs(bc[:,boundary[:,4]==0,2:4])))
    divergence=float(np.max(np.abs(dx[...,0]+dy[...,1] if projected else dx[...,2]+dy[...,3])))
    if wall_gap>2e-5:raise ValueError(f'Hard no-slip constraint failed: {wall_gap}')
    if 'velocity' in c and max(flux_gap,wall_gap,divergence)>2e-5:
        raise ValueError(f'Hard flow constraint failed: flux={flux_gap}, wall={wall_gap}, div={divergence}')
    full=combined[:,slices[4]]
    area=np.sum((1-full[...,0])/2*np.asarray(data['volumes'])[None],axis=1)
    b,t=walls(saved['points'][:,0],saved['amplitudes'],c,xp=np);measure=t-b
    def relative(a,b,weighted):
        if weighted:
            w=measure.reshape((1,len(measure))+(1,)*(a.ndim-2))
        else:w=1.
        return float(np.sqrt(np.sum((a-b)**2*w)/np.sum(b*b*w)))
    metrics={}
    for label,sl in [('phi',slice(0,1)),('velocity',slice(2,4)),('pressure',slice(4,5))]:
        start=0 if label=='phi' else 1
        a,b=replay[start:,...,sl],saved['reference'][start:,...,sl]
        metrics[label]={kind:relative(a,b,weighted) for kind,weighted in [('unweighted',False),('physical_area_weighted',True)]}
        if abs(metrics[label]['unweighted']-summary['metrics'][label])>2e-6:
            raise ValueError(f'Logged {label} metric does not replay')
    diagnostic=run/'diagnostic_fields.npz'
    np.savez_compressed(diagnostic,fields=full,points=np.asarray(data['points']),
        time=saved['time'],amplitudes=saved['amplitudes'])
    report=dict(status='replay_and_hard_constraints_passed',run=str(run.resolve()),
        continuity_enforcement='hard_streamfunction' if 'velocity' in c else 'soft_PDE_residual',
        soft_continuity_accuracy_pass_claimed=False,
        saved_field_max_gap=gap,primal_jvp_max_gap=primal_gap,
        primal_query_batching='Independent saved/probe/section/boundary/domain queries concatenated; separate time-JVP probe',
        maximum_flux_error=flux_gap,maximum_wall_speed=wall_gap,maximum_divergence=divergence,
        section_x=section.tolist(),flux_times=np.asarray(probe_time).tolist(),flux=flux.tolist(),
        phase_area_times=saved['time'].tolist(),phase_area=area.tolist(),
        full_grid_phase_min=full[...,0].min(axis=1).tolist(),
        full_grid_phase_max=full[...,0].max(axis=1).tolist(),
        negative_phase_area=np.sum((full[...,0]<0)*np.asarray(data['volumes'])[None],axis=1).tolist(),
        negative_phase_area_scope='Zero-contour core diagnostic, not the conserved diffuse-phase integral',
        full_grid_diagnostic_fields=str(diagnostic.resolve()),full_grid_points=len(data['points']),
        phase_area_scope='Diagnostic only: open-domain mass balance includes boundary flux, not imposed constant mass',
        condition_visit_counts=np.bincount(indices.ravel(),minlength=n).tolist(),
        sampling_start_step=start_step,sampling_updates=updates,
        actual_draws_reproduced_from_global_keys=True,
        actual_time_strata_passed=True,supervised_evolved_field_samples=0,metrics=metrics,
        diagnostic_queries=len(saved['points']),
        scope='One development FEM geometry; no population estimate')
    if online_audit is not None:report['online_inputs']=online_audit
    if 'time_derivative' in c:
        report['time_derivative']=dict(**c.time_derivative.to_dict(),
            actual_time_samples=c.sample.Nt,updates=updates,
            resampled_every_update=True,shared_across_condition_batch=True,
            scope='Exact saved draws reproduced from global keys and checked against continuous strata; no fixed BDF node indices')
    if 'derivatives' in c:
        from examples.channel_nsch2d.losses import gmls_first_jets
        domain=jnp.asarray(full,dtype=dtype)
        gx,gy=gmls_first_jets(domain,domain,data,boundary=False)
        div=np.asarray(gx[...,2]+gy[...,3],dtype=np.float64)
        mask=np.asarray(data['interior'])>0
        integral=c.nl.method=='integral_moment'
        key='integral_residual_derivatives'if integral else 'GMLS_residual_derivatives'
        spatial='integral'if integral else 'GMLS'
        report[key]=dict(configuration=c.derivatives.to_dict(),
            maximum_interior_discrete_divergence=float(np.max(np.abs(div[:,mask]))),
            scope='Approximate '+spatial+' divergence is diagnostic, not the exact physical AD-curl constraint or a training loss')
    if c.nl.method == 'fem_weak_p2p1':
        cells=np.asarray(data['fem_cells']);gradient=np.asarray(data['fem_gradient'],dtype=np.float64)
        measure=np.asarray(data['fem_measure'],dtype=np.float64)
        div=np.einsum('teic,eqic->teq',full[:,cells,2:4],gradient)
        report['FE_interpolant_divergence']=dict(maximum=float(np.max(np.abs(div))),
            physical_rms_by_time=np.sqrt(np.sum(div**2*measure[None],axis=(1,2))/measure.sum()).tolist(),
            scope='P2 velocity interpolant at FE quadrature points; distinct from exact neural AD-curl incompressibility')
    if refinement_audit is not None:
        report['refinement_inputs']=refinement_audit
        report['sampling_scope']='Inherited parent stochastic stage; new refinement uses the audited complete BDF time grid'
    if initialization_audit is not None:report['model_initialization']=initialization_audit
    if projected:
        report['chemical_constraint']=dict(configuration=c.chemical_constraint.to_dict(),
            saved_query_rule_reproduced=True,physical_mu='Projected interior nodal values; retained boundary decoder traces; explicit mapped interpolation at other queries',
            spatial_mu_AD_claimed=False,redundant_chemical_soft_loss=False)
    (run/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    validate(parser.parse_args().run)
