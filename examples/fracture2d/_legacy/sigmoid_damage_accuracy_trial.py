"""Isolated fixed-H training controls and same-load sigmoid stagger continuation."""
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
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d._legacy.asymmetric_damage_range import evaluate_asymmetric
from examples.fracture2d._legacy.bounded_damage_morphology_trial import make_problem, morphology
from examples.fracture2d._legacy.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.core_halo_adam import make_adam
from examples.fracture2d._legacy.core_halo_damage import (
    crack_sampling_probabilities, pack_rows, predict_rows, sampled_power_loss,
    validate_probabilities,
)
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_residual_weighting import build_core_halo, core_matvec
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network, map_damage
from examples.fracture2d._legacy.provisional_damage import crack_extent, mesh_adjacency
from examples.fracture2d._legacy.range_damage_optimization import damage_diagnostics
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def make_sampled_sigmoid_loss(*, gc0, gc1, gain, power, floor, population_size):
    """Apply the bounded head at every halo node before the original CSR rows."""
    def halo_image(net, context):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        return map_damage('sigmoid', predict_rows(net, c['features'],
            c['basis_columns'], c['basis_values'], trunk_gain=gain))

    def loss(net, context, load):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        halo = halo_image(net, c)
        d = halo[c['core_local']]
        residual = (gc0+2*c['core_H'])*d+gc1*core_matvec(
            halo, c['operator_columns'], c['operator_values'])-2*c['core_H']
        return sampled_power_loss(residual, c['H'], c['p_core'],
            population_size=population_size, power=power, floor=floor)
    return halo_image, loss


def sampled_context(p, rng, probabilities, *, count, capacity):
    core = rng.choice(p.n, size=count, replace=True, p=probabilities).astype(np.int32)
    packed = build_core_halo(p.D, core, node_capacity=capacity, row_width=p.row_width)
    ids = packed['global_ids']
    context = dict(H=p.H, core_H=p.H[core], features=p.X[ids],
        basis_columns=p.bc[ids], basis_values=p.bv[ids],
        core_local=jnp.asarray(packed['core_local']),
        operator_columns=jnp.asarray(packed['columns']),
        operator_values=jnp.asarray(packed['values']),
        p_core=jnp.asarray(probabilities[core]))
    stats = dict(actual_halo=int(packed['node_mask'].sum()),
        unique_core=int(len(np.unique(core))), core_count=count, padded_halo=capacity)
    return context, core, stats


def prepare(c):
    a = c.accuracy
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform != 'gpu'
            or a.index != 53 or a.source_sweep != 128 or c.stagger.omega != .5
            or a.outer_max <= a.source_sweep or a.outer_max > c.outer_stop.max_sweeps
            or not 0 < a.residual_limit <= c.outer_stop.inner_AT2_relative_limit
            or list(a.adam_rates) != [1e-4, 1e-5]
            or any(type(x) is not int or x < 1 for x in [a.full_short, a.full_long,
                a.adam_iterations, a.polish_iterations, a.inner_iterations,
                a.core_count, a.halo_capacity, a.resample_every, a.check_every])):
        raise ValueError('accuracy trial requires explicit GPU/x64 critical-step controls')
    root = Path(a.source_root)
    for name, digest in a.pins.items():
        if sha(root/name) != digest:
            raise ValueError(f'accuracy source hash changed: {name}')
    identity = json.loads((root/'identity.json').read_text())
    expected = c.to_dict()
    del expected['accuracy']
    if expected != identity['config'] or not identity['direct_loading']:
        raise ValueError('accuracy trial changes the bounded parent configuration')
    for name, digest in identity['code'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'protected source changed: {name}')
    for path, digest in identity['source'].items():
        if sha(path) != digest:
            raise ValueError(f'protected ancestry changed: {path}')
    audit = json.loads((root/a.audit_relative_path).read_text())
    if not audit['pass_replay'] or not any(r['head'] == 'sigmoid' and r['index'] == 53
            for r in audit['coupled_results']):
        raise ValueError('accuracy trial requires independent sigmoid53 replay')
    source = root/'sigmoid/state_0053'
    report = json.loads((source/'result.json').read_text())
    if (report['index'] != 53 or report['sweeps'] != a.source_sweep
            or report['state_sha256'] != sha(source/'fields.npz')
            or report['network_sha256'] != sha(source/'network.eqx')
            or report['parent_sha256'] != c.head.source52_sha256):
        raise ValueError('accuracy source does not preserve committed52 ancestry')
    m, operator, geometry = load_geometry(c)
    if geometry != identity['geometry'] or sha(c.cold.basis_path) != c.cold.basis_sha256:
        raise ValueError('accuracy mesh/operator/basis identity differs')
    B, D = load_npz(c.cold.basis_path), operator.to_scipy()
    n, xy, V = m.n_material, m.coordinates[:m.n_material], m.volumes
    with np.load(source/'fields.npz', allow_pickle=False) as z:
        saved = {k:z[k].copy() for k in z.files}
    with np.load(c.head.source52, allow_pickle=False) as z:
        parent = {k:z[k].copy() for k in z.files}
    for name in ['d', 'H', 'H_used', 'damage_image', 'd_parent', 'H_parent']:
        if saved[name].shape != (n,) or saved[name].dtype != np.float64 or not np.isfinite(saved[name]).all():
            raise ValueError(f'invalid ordered source array: {name}')
    if (saved['d'].min() < 0 or saved['d'].max() > 1 or saved['H'].min() < 0
            or saved['H'].max() <= 0 or saved['q'].shape != m.load_direction.shape
            or not np.isfinite(saved['q']).all()):
        raise ValueError('invalid physical sigmoid source range/history/displacement')
    np.testing.assert_array_equal(saved['d_parent'], parent['d'])
    np.testing.assert_array_equal(saved['H_parent'], parent['H'])
    np.testing.assert_array_equal(saved['H'], np.maximum(parent['H'], m.driver(saved['q'])))
    load = jnp.asarray(a.index*c.data.load_step, jnp.float64)
    np.testing.assert_array_equal(saved['q'][m.constrained], float(load)*m.load_direction[m.constrained])
    force = m.matrix(saved['d'])@saved['q']
    row = json.loads((source/f'sweep_{a.source_sweep:04d}.json').read_text())
    if np.linalg.norm(force[m.free]) > row['mechanics']['convergence_limit']*(1+1e-3):
        raise ValueError('source128 independent mechanics residual fails')
    np.testing.assert_allclose(force@m.load_direction, report['reaction'], rtol=1e-9, atol=1e-6)
    with np.load(root/'fixed_0052/sigmoid/parameterization.npz') as z:
        scales, gain = z['column_scale'].copy(), float(z['trunk_gain'])
    bc, bv = pack_rows(B@diags(scales), width=int(np.max(np.diff(B.indptr))))
    bc, bv = jnp.asarray(bc), jnp.asarray(bv)
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    feature, info = make_features(operator, xy, gc0=g0, length=c.physical.l0,
        tip=list(c.conditioned.tip), powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    feature = eqx.filter_jit(feature)
    def context(h):
        h = jax.lax.stop_gradient(jnp.asarray(h, jnp.float64))
        return dict(H=h, features=jax.lax.stop_gradient(feature(h)))
    matvec, packing = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    image, objective, metrics = make_problem(bc, bv, matvec, head='sigmoid',
        gain=gain, gc0=g0, gc1=g1, power=c.core.power, floor=c.conditioned.loss_floor,
        volumes=jnp.asarray(V), thresholds=tuple(c.range.activity_thresholds))
    initial = eqx.tree_deserialise_leaves(source/'network.eqx',
        fresh_network(c, info['input_dimension'], B, head='sigmoid'))
    image, metrics = eqx.filter_jit(image), eqx.filter_jit(metrics)
    np.testing.assert_allclose(image(initial, context(saved['H_used']), load),
        saved['damage_image'], rtol=1e-10, atol=1e-12)
    H = jnp.asarray(saved['H'])
    halo_image, sampled_loss = make_sampled_sigmoid_loss(gc0=g0, gc1=g1, gain=gain,
        power=c.core.power, floor=c.conditioned.loss_floor, population_size=n)
    return SimpleNamespace(m=m, D=D, geometry=geometry, n=n, points=xy, V=V,
        saved=saved, parent=parent, source_report=report, source_row=row, initial=initial,
        load=load, H=H, X=context(H)['features'], bc=bc, bv=bv, gain=gain,
        row_width=packing['row_width']+1, context=context, image=image, metrics=metrics,
        objective=objective, halo_image=eqx.filter_jit(halo_image), sampled_loss=sampled_loss,
        matvec=matvec, g0=g0, g1=g1)


def run(c, *, preflight_only):
    a, p = c.accuracy, prepare(c)
    ctx = p.context(p.H)
    def measure(net):
        values = {k:float(v) for k,v in p.metrics(net, ctx, p.load).items()}
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError('nonfinite full-domain accuracy metrics')
        return values
    probabilities = np.asarray(crack_sampling_probabilities(jnp.asarray(p.points),
        start=c.core.segment_start, end=c.core.segment_end, width=c.core.band_width,
        boost=c.core.boost, power=c.core.density_power))
    validate_probabilities(probabilities, population_size=p.n)
    if a.halo_capacity > p.n:
        raise ValueError('halo capacity exceeds physical population')
    sample, core, pool = sampled_context(p, np.random.default_rng(c.core.seed), probabilities,
        count=a.core_count, capacity=a.halo_capacity)
    full_image = np.asarray(p.image(p.initial, ctx, p.load))
    halo = p.halo_image(p.initial, sample)
    np.testing.assert_allclose(np.asarray(halo)[np.asarray(sample['core_local'])], full_image[core],
        rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(core_matvec(halo, sample['operator_columns'], sample['operator_values']),
        (p.D@full_image)[core], rtol=1e-10, atol=1e-8)
    print(json.dumps(dict(event='preflight_pass', source_sweep=a.source_sweep,
        source_reaction=p.source_report['reaction'], initial_own_H=measure(p.initial),
        exact_core_halo=True, pool=pool)), flush=True)
    if preflight_only:
        return
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    codes = set(json.loads((Path(a.source_root)/'identity.json').read_text())['code'])
    codes.update(['sigmoid_damage_accuracy_trial.py', 'core_halo_adam.py', 'damage_residual_weighting.py'])
    (root/'code_snapshot').mkdir()
    for name in sorted(codes):
        shutil.copyfile(Path(__file__).with_name(name), root/'code_snapshot'/name)
    code = {name:sha(root/'code_snapshot'/name) for name in sorted(codes)}
    shutil.copyfile(Path(a.source_root)/a.audit_relative_path, root/'source_audit.json')
    atomic_json(root/'identity.json', dict(config=c.to_dict(), source_pins=a.pins.to_dict(),
        code=code, geometry=p.geometry, committed_parent_index=52, fixed_load_index=53,
        initial_sweep=128, head='sigmoid', direct_training_targets=False,
        physical_state_promoted_by_fixed_fits=False, detached_history_and_probabilities=True,
        exact_original_support=True, importance_corrected_moment=True,
        nonlinear_sampled_root_unbiased=False, adam_memory_persistent_between_resamples=True))
    np.save(root/'sampling_probabilities.npy', probabilities)
    reports, candidates, continuation = [], [], dict(complete=False, status='not_started')
    def publish(status, complete):
        atomic_json(root/'result.json', dict(status=status, complete=complete,
            fixed_reports=reports, continuation=continuation))
    publish('fixed_history_training', False)
    fit, value_grad, evaluate = make_cached_lbfgs(p.initial, p.objective, p.metrics)
    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(eqx.filter(p.initial, eqx.is_inexact_array))
    jax.block_until_ready((value_grad(flat, ctx, p.load), evaluate(flat, ctx, p.load)))
    def controls(iterations):
        values = c.head.lbfgs.to_dict()
        values.update(maxiter=iterations, check_every=a.check_every, residual_limit=a.residual_limit)
        return values
    def save_case(name, net, *, seconds, optimization, extra):
        folder = root/name
        folder.mkdir()
        values = measure(net)
        damage = np.asarray(p.image(net, ctx, p.load))
        atomic_fields(folder/'fields.npz', d=damage, H=np.asarray(p.H))
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        report = dict(name=name, seconds=seconds, metrics=values, optimization=optimization,
            fields_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
            same_initial_network=True, same_H=True, physical_state_promoted=False, **extra)
        atomic_json(folder/'result.json', report)
        reports.append(report)
        candidates.append((net, report))
        publish('fixed_history_training', False)
        print(json.dumps(dict(event='fixed_case_complete', name=name, seconds=seconds,
            AT2=values['AT2_relative'])), flush=True)
        return folder
    save_case('initial', p.initial, seconds=0., optimization=[], extra=dict(sampling_used=False))
    for name, budget in [('full_512', a.full_short), ('full_extended', a.full_long)]:
        started = time.perf_counter()
        net, opt = fit(p.initial, ctx, p.load, controls(budget), progress=lambda r:
            print(json.dumps(dict(event='fixed_lbfgs', name=name, **r)), flush=True))
        save_case(name, net, seconds=time.perf_counter()-started, optimization=[opt],
            extra=dict(sampling_used=False, maximum_iterations=budget))
    s = c.core.adam
    for rate in a.adam_rates:
        name = f'sampled_adam_{rate:g}'
        optimizer, step, schedule = make_adam(p.sampled_loss, learning_rate=rate,
            decay_steps=a.adam_iterations, decay_rate=s.decay_rate,
            minimum_rate=rate*s.minimum_fraction, b1=s.b1, b2=s.b2,
            eps=s.eps, eps_root=s.eps_root, grad_clip=s.grad_clip)
        net = p.initial
        state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
        jax.block_until_ready(step(net, state, sample, p.load))
        rng = np.random.default_rng(c.core.seed)
        best, best_metrics, best_iteration = net, measure(net), 0
        trace = [dict(iteration=0, seconds=0., **best_metrics)]
        cores, pools = [], []
        started = time.perf_counter()
        for iteration in range(1, a.adam_iterations+1):
            if (iteration-1) % a.resample_every == 0:
                current, ids, stats = sampled_context(p, rng, probabilities,
                    count=a.core_count, capacity=a.halo_capacity)
                cores.append(ids)
                pools.append(dict(first_iteration=iteration, **stats))
            net, state, value, gradient_norm = step(net, state, current, p.load)
            if iteration % a.check_every == 0 or iteration == a.adam_iterations:
                metrics = measure(net)
                if not np.isfinite([float(value), float(gradient_norm)]).all():
                    raise FloatingPointError('nonfinite sampled Adam loss/gradient')
                if metrics['AT2_relative'] < best_metrics['AT2_relative']:
                    best, best_metrics, best_iteration = net, metrics, iteration
                row = dict(iteration=iteration, seconds=time.perf_counter()-started,
                    sampled_loss=float(value), gradient_norm=float(gradient_norm), **metrics)
                trace.append(row)
                print(json.dumps(dict(event='fixed_adam', name=name, **row)), flush=True)
                if metrics['AT2_relative'] <= a.residual_limit:
                    break
        jax.block_until_ready(state)
        adam_seconds = time.perf_counter()-started
        count = int(state[1][0].count)
        if count != iteration:
            raise RuntimeError('Adam memory/count reset during resampling')
        polished, opt = fit(best, ctx, p.load, controls(a.polish_iterations), progress=lambda r:
            print(json.dumps(dict(event='fixed_polish', name=name, **r)), flush=True))
        # Keep the best global check if polishing ends with a worse infinity metric.
        if measure(polished)['AT2_relative'] > best_metrics['AT2_relative']:
            selected = best
        else:
            selected = polished
        folder = save_case(name, selected, seconds=time.perf_counter()-started, optimization=[opt],
            extra=dict(sampling_used=True, learning_rate=rate, adam_iterations=iteration,
                adam_count=count, best_adam_iteration=best_iteration, adam_metrics=best_metrics,
                adam_seconds=adam_seconds, trace=trace, pools=pools, fresh_draw_blocks=len(cores)))
        np.save(folder/'core_samples.npy', np.asarray(cores, np.int32))
        eqx.tree_serialise_leaves(folder/'last_adam_network.eqx', net)
        eqx.tree_serialise_leaves(folder/'last_adam_state.eqx', state)
    net, chosen = min(candidates, key=lambda item:(item[1]['metrics']['AT2_relative'], item[1]['seconds']))
    atomic_json(root/'selection.json', dict(name=chosen['name'], metric='full_domain_AT2_relative',
        value=chosen['metrics']['AT2_relative'], network_sha256=chosen['network_sha256'],
        physical128_unchanged=True, committed_history_index=52))
    print(json.dumps(dict(event='training_selected', name=chosen['name'],
        AT2=chosen['metrics']['AT2_relative'])), flush=True)
    branch = root/'state_0053'
    branch.mkdir()
    q, d, H = [jnp.asarray(p.saved[k]) for k in ['q', 'd', 'H']]
    dp, hp = [jnp.asarray(p.parent[k]) for k in ['d', 'H']]
    reaction = p.source_report['reaction']
    solve = p.m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
        maxiter=c.validation.cg_maxiter)
    field_metrics = eqx.filter_jit(lambda damage, history:damage_diagnostics(
        damage, history, p.matvec, gc0=p.g0, gc1=p.g1, volumes=jnp.asarray(p.V),
        thresholds=tuple(c.range.activity_thresholds)))
    graph = mesh_adjacency(p.m.triangles[:p.m.n_material_cells], population_size=p.n)
    count = p.source_row['outer_convergence']['consecutive']
    continuation.update(status='running', selected_training=chosen['name'], sweep=a.source_sweep,
        reaction=reaction, local_strict_ready=False, load_completed=False)
    publish('same_load_continuation', False)
    started = time.perf_counter()
    for sweep in range(a.source_sweep+1, a.outer_max+1):
        oldq, oldd, oldh, oldr = q, d, H, reaction
        hu = jax.lax.stop_gradient(H)
        current = p.context(hu)
        net, opt = fit(net, current, p.load, controls(a.inner_iterations), progress=lambda r:
            print(json.dumps(dict(event='accuracy_inner', sweep=sweep, **r)), flush=True))
        image = p.image(net, current, p.load)
        inner = {k:float(v) for k,v in p.metrics(net, current, p.load).items()}
        d = oldd+c.stagger.omega*(image-oldd)
        q, reaction, mechanical = solve(d, p.load, q)
        H = jnp.maximum(hp, jnp.asarray(p.m.driver(np.asarray(q))))
        physical = {k:float(v) for k,v in field_metrics(d, H).items()}
        raw = {k:float(v) for k,v in p.metrics(net, p.context(H), p.load).items()}
        if not np.isfinite(list(physical.values())+list(inner.values())+list(raw.values())).all():
            raise FloatingPointError('nonfinite accuracy staggered state')
        row = dict(index=a.index, load=float(p.load), sweep=sweep, reaction=reaction,
            damage_min=physical['damage_min'], damage_max=physical['damage_max'],
            fixed_H_image_damage_min=inner['damage_min'], fixed_H_image_damage_max=inner['damage_max'],
            fixed_H_image_AT2_relative=inner['AT2_relative'], coupled_AT2_relative=physical['AT2_relative'],
            raw_current_H_AT2_relative=raw['AT2_relative'], damage_change=float(jnp.max(abs(d-oldd))),
            q_change_over_load=float(jnp.max(abs(q-oldq)))/float(p.load),
            H_change_relative=float(jnp.max(abs(H-oldh)))/max(float(H.max()),float(oldh.max()),c.outer_stop.scale_floor),
            reaction_change_relative=abs(reaction-oldr)/max(abs(reaction),abs(oldr),c.outer_stop.scale_floor),
            healing=float(jnp.max(jnp.maximum(dp-d,0))), history_decrease=float(jnp.max(jnp.maximum(hp-H,0))),
            drift_count=int(np.count_nonzero(np.linalg.norm(np.asarray(q).reshape(-1,2)[:p.n],axis=1)/float(p.load)>1.08)),
            mechanics=mechanical, parent_sha256=c.head.source52_sha256, inner_optimization=opt,
            morphology=morphology(d,p.points,p.V,band_width=c.head.background_band_lengths*c.physical.l0,
                threshold=c.head.background_threshold), physical_diagnostics=physical)
        outer = evaluate_asymmetric(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count,
            negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
        outer['checks']['raw_current_H_AT2'] = dict(value=raw['AT2_relative'],
            limit=c.reporting.coupled_AT2_relative_limit,
            **{'pass':raw['AT2_relative'] <= c.reporting.coupled_AT2_relative_limit})
        passed = all(v['pass'] for v in outer['checks'].values())
        count = count+1 if passed else 0
        outer.update(pass_this_sweep=passed,consecutive=count,
            converged=sweep>=c.outer_stop.min_sweeps and count>=c.outer_stop.consecutive_passes)
        row['outer_convergence'] = outer
        atomic_json(branch/f'sweep_{sweep:04d}.json', row)
        atomic_fields(branch/'pending_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),
            H_used=np.asarray(hu),damage_image=np.asarray(image),d_parent=np.asarray(dp),H_parent=np.asarray(hp),
            d_previous=np.asarray(oldd),q_previous=np.asarray(oldq),H_previous=np.asarray(oldh))
        eqx.tree_serialise_leaves(branch/'pending_network.eqx', net)
        pending = dict(index=a.index,load=float(p.load),sweep=sweep,reaction=reaction,
            fields_sha256=sha(branch/'pending_fields.npz'), network_sha256=sha(branch/'pending_network.eqx'))
        atomic_json(branch/'pending.json', pending)
        if sweep in a.checkpoints or sweep == a.source_sweep+1 or outer['converged']:
            checkpoint = branch/f'checkpoint_{sweep:04d}'
            checkpoint.mkdir()
            shutil.copyfile(branch/'pending_fields.npz',checkpoint/'fields.npz')
            shutil.copyfile(branch/'pending_network.eqx',checkpoint/'network.eqx')
            atomic_json(checkpoint/'report.json',dict(**row,fields_sha256=sha(checkpoint/'fields.npz'),
                network_sha256=sha(checkpoint/'network.eqx')))
        continuation.update(physical_AT2=physical['AT2_relative'],
            fixed_H_AT2=inner['AT2_relative'],raw_current_H_AT2=raw['AT2_relative'],
            local_strict_ready=bool(outer['converged']),seconds=time.perf_counter()-started,
            morphology=row['morphology'],outer_convergence=outer,**pending)
        publish('same_load_continuation',False)
        print(json.dumps(dict(event='accuracy_outer',sweep=sweep,reaction=reaction,
            inner_AT2=inner['AT2_relative'],physical_AT2=physical['AT2_relative'],
            raw_AT2=raw['AT2_relative'],outside_mean=row['morphology']['outside_band_mean'])),flush=True)
        if outer['converged']:
            break
    force = p.m.matrix(np.asarray(d))@np.asarray(q)
    if np.linalg.norm(force[p.m.free]) > mechanical['convergence_limit']*(1+1e-3):
        raise RuntimeError('final independent host FEM force check failed')
    np.testing.assert_allclose(force@p.m.load_direction,reaction,rtol=1e-9,atol=1e-6)
    continuation.update(complete=True,status='ready' if outer['converged'] else 'budget_exhausted',
        load_completed=bool(outer['converged']),crack=crack_extent(d,p.points,graph,threshold=.9,
            tip=[.5,.5],tip_radius=.024,right_edge=1.,tolerance=1e-12))
    for name,digest in a.pins.items():
        if sha(Path(a.source_root)/name) != digest:
            raise RuntimeError(f'protected source changed during accuracy trial: {name}')
    for name,digest in code.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise RuntimeError(f'accuracy numerical source changed: {name}')
    publish('complete',True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config'](),preflight_only=args.preflight)
