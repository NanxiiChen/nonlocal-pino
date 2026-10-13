"""Quarter-load staggered sigmoid PINN control with explicit provisional ancestry."""
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

from examples.fracture2d.asymmetric_damage_range import evaluate_asymmetric
from examples.fracture2d.bounded_damage_morphology_trial import make_problem, morphology
from examples.fracture2d.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d.conditioned_damage import make_features
from examples.fracture2d.core_halo_damage import pack_rows
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import load_geometry, sha
from examples.fracture2d.output_damage_continue import atomic_fields
from examples.fracture2d.plain_damage_head import fresh_network
from examples.fracture2d.provisional_damage import advancement, crack_extent, mesh_adjacency
from examples.fracture2d.range_damage_optimization import damage_diagnostics
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def load_schedule(indices, step, *, initial_load, end_load):
    ids = np.asarray(indices)
    if (ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer)
            or not np.isfinite([step, initial_load, end_load]).all() or step <= 0
            or initial_load <= 0 or end_load <= initial_load or np.any(np.diff(ids) != 1)):
        raise ValueError('refined schedule requires consecutive integer IDs and finite positive increments')
    loads = ids.astype(np.float64)*step
    if (loads[0] <= initial_load or not np.isclose(loads[0]-initial_load, step, rtol=1e-12, atol=1e-15)
            or not np.isclose(loads[-1], end_load, rtol=1e-12, atol=1e-15)):
        raise ValueError('refined loads do not connect the declared physical parent and endpoint')
    return loads


def committed_history(previous_load_history, current_driver):
    """Use the preceding load's history; never accumulate an intermediate sweep."""
    return jax.lax.stop_gradient(jnp.maximum(previous_load_history, current_driver))


def prepare(c):
    a, root = c.refinement, Path(c.refinement.source_root)
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform != 'gpu'
            or c.stagger.omega != .5 or a.head != 'sigmoid' or a.parent_index != 52
            or a.outer_iterations > c.outer_stop.max_sweeps
            or any(type(v) is not int or v < 1 for v in
                [a.outer_iterations, a.inner_iterations, a.check_every])
            or type(a.user_provisional) is not bool):
        raise ValueError('refined sigmoid requires explicit GPU/x64 budgets and loading policy')
    for name, digest in a.pins.items():
        if sha(root/name) != digest:
            raise ValueError(f'changed refined parent source: {name}')
    identity = json.loads((root/'identity.json').read_text())
    expected = c.to_dict()
    del expected['refinement']
    if expected != identity['config']:
        raise ValueError('refined branch changes the bounded parent model/configuration')
    for path, digest in identity['source'].items():
        if sha(path) != digest:
            raise ValueError(f'changed bounded ancestry: {path}')
    for name, digest in identity['code'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'protected source changed: {name}')
    if sha(c.head.source52) != c.head.source52_sha256:
        raise ValueError('physical52 parent hash differs')
    if sha(c.core.restart.source_report) != c.core.restart.source_report_sha256:
        raise ValueError('physical52 report hash differs')
    report = json.loads(Path(c.core.restart.source_report).read_text())
    if (report['status'] != 'ready' or report['index'] != 52
            or report['state_sha256'] != c.head.source52_sha256):
        raise ValueError('refined branch requires certified physical52')
    audit = json.loads((root/a.audit_relative_path).read_text())
    if not audit['pass_replay'] or not any(r['head'] == 'sigmoid' and r['index'] == 52
            for r in audit['fixed_results']):
        raise ValueError('refined branch needs independent sigmoid52 native-head replay')
    loads = load_schedule(list(a.load_ids), a.load_step,
        initial_load=report['load'], end_load=a.end_load)
    m, operator, geometry = load_geometry(c)
    if geometry != identity['geometry'] or sha(c.cold.basis_path) != c.cold.basis_sha256:
        raise ValueError('refined mesh/operator/basis differs from bounded parent')
    with np.load(c.head.source52, allow_pickle=False) as z:
        parent = {k:z[k].copy() for k in z.files}
    if (parent['d'].shape != (m.n_material,) or parent['H'].shape != parent['d'].shape
            or parent['q'].shape != m.load_direction.shape or parent['d'].min() < 0
            or parent['d'].max() > 1 or parent['H'].min() < 0 or parent['H'].max() <= 0
            or any(z.dtype != np.float64 or not np.isfinite(z).all() for z in parent.values())):
        raise ValueError('invalid physical52 arrays/ranges/order')
    np.testing.assert_array_equal(parent['H'], np.maximum(parent['H_parent'], m.driver(parent['q'])))
    np.testing.assert_array_equal(parent['q'][m.constrained],report['load']*m.load_direction[m.constrained])
    force = m.matrix(parent['d'])@parent['q']
    if np.linalg.norm(force[m.free]) > report['outer_convergence']['checks']['mechanics']['limit']*(1+1e-3):
        raise ValueError('independent physical52 mechanical equilibrium fails')
    np.testing.assert_allclose(force@m.load_direction,report['reaction'],rtol=1e-9,atol=1e-6)
    B = load_npz(c.cold.basis_path)
    folder = root/'fixed_0052/sigmoid'
    with np.load(folder/'parameterization.npz') as z:
        scales, gain = z['column_scale'].copy(), float(z['trunk_gain'])
    bc, bv = pack_rows(B@diags(scales), width=int(np.max(np.diff(B.indptr))))
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    xy, V = m.coordinates[:m.n_material], m.volumes
    feature, info = make_features(operator,xy,gc0=g0,length=c.physical.l0,
        tip=list(c.conditioned.tip),powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    feature = eqx.filter_jit(feature)
    def context(h):
        h = jax.lax.stop_gradient(jnp.asarray(h,jnp.float64))
        return dict(H=h,features=jax.lax.stop_gradient(feature(h)))
    matvec = jax.jit(make_padded_matvec(operator)[0])
    image, loss, metrics = make_problem(jnp.asarray(bc),jnp.asarray(bv),matvec,
        head='sigmoid',gain=gain,gc0=g0,gc1=g1,power=c.core.power,
        floor=c.conditioned.loss_floor,volumes=jnp.asarray(V),thresholds=tuple(c.range.activity_thresholds))
    image, metrics = eqx.filter_jit(image), eqx.filter_jit(metrics)
    net = eqx.tree_deserialise_leaves(folder/'network.eqx',fresh_network(c,info['input_dimension'],B,head='sigmoid'))
    with np.load(folder/'fields.npz') as z:
        np.testing.assert_array_equal(z['H'],parent['H'])
        np.testing.assert_allclose(image(net,context(parent['H']),jnp.asarray(report['load'])),
            z['d'],rtol=1e-10,atol=1e-12)
    return SimpleNamespace(m=m,operator=operator,geometry=geometry,parent=parent,parent_report=report,
        net=net,image=image,loss=loss,metrics=metrics,context=context,matvec=matvec,
        g0=g0,g1=g1,points=xy,V=V,loads=loads,parent_identity=identity)


def run(c, *, preflight_only):
    a, p = c.refinement, prepare(c)
    load52 = jnp.asarray(p.parent_report['load'],jnp.float64)
    initial = {k:float(v) for k,v in p.metrics(p.net,p.context(p.parent['H']),load52).items()}
    print(json.dumps(dict(event='refined_preflight_pass',loads=p.loads.tolist(),
        physical_parent_index=52,physical_reaction=p.parent_report['reaction'],network_metrics=initial)),flush=True)
    if preflight_only:
        return
    root = Path(a.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    names = set(p.parent_identity['code'])|{'sigmoid_damage_load_refined.py','provisional_damage.py',
        'range_damage_optimization.py','split_pinn_gradients.py'}
    (root/'code_snapshot').mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    codes = {name:sha(root/'code_snapshot'/name) for name in sorted(names)}
    shutil.copyfile(Path(a.source_root)/a.audit_relative_path,root/'source_audit.json')
    atomic_json(root/'identity.json',dict(config=c.to_dict(),source_pins=a.pins.to_dict(),
        code=codes,geometry=p.geometry,source_physical52_sha256=c.head.source52_sha256,
        loads=p.loads.tolist(),load_increment=a.load_step,physical_parent_index=52,head='sigmoid',
        warm_same_head52_only=True,old_identity_weights=False,direct_training_targets=False,
        own_zero_bounded_path=False,physical_parent_certified=True,network52_certified=False,
        unchanged_per_load_gates=True,per_step_caps_can_increase_total_work=True))
    curve, current = [], dict(status='not_started')
    def publish(complete,status):
        atomic_json(root/'result.json',dict(complete=complete,status=status,curve=curve,current=current,
            strict_loads=sum(r['local_strict_ready'] for r in curve),total_outer_updates=sum(r['sweeps'] for r in curve),
            total_inner_updates=sum(r['inner_updates'] for r in curve)))
    net = p.net
    fit = make_cached_lbfgs(net,p.loss,p.metrics)[0]
    controls = c.head.lbfgs.to_dict()
    controls.update(maxiter=a.inner_iterations,check_every=a.check_every)
    solve = p.m.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,maxiter=c.validation.cg_maxiter)
    physical_metrics = eqx.filter_jit(lambda d,h:damage_diagnostics(d,h,p.matvec,gc0=p.g0,gc1=p.g1,
        volumes=jnp.asarray(p.V),thresholds=tuple(c.range.activity_thresholds)))
    graph = mesh_adjacency(p.m.triangles[:p.m.n_material_cells],population_size=p.m.n_material)
    q,d,H = [jnp.asarray(p.parent[k]) for k in ['q','d','H']]
    parent_sha,ancestry = c.head.source52_sha256,False
    publish(False,'running')
    for ordinal,load_value in enumerate(p.loads,1):
        index = a.parent_index+ordinal
        folder = root/f'state_{index:04d}'
        folder.mkdir()
        load = jnp.asarray(load_value,jnp.float64)
        dp,hp = jax.tree.map(jax.lax.stop_gradient,(d,H))
        q,reaction,mechanical = solve(d,load,q)
        H = committed_history(hp,jnp.asarray(p.m.driver(np.asarray(q))))
        count,inner_total = 0,0
        started = time.perf_counter()
        current = dict(index=index,load=float(load),load_id=int(a.load_ids[ordinal-1]),sweep=0,
            status='running',parent_sha256=parent_sha,reaction=reaction)
        publish(False,'running')
        for sweep in range(1,a.outer_iterations+1):
            oldq,oldd,oldh,oldr = q,d,H,reaction
            hu = jax.lax.stop_gradient(H)
            context = p.context(hu)
            net,opt = fit(net,context,load,controls,progress=lambda r:print(json.dumps(
                dict(event='refined_inner',index=index,load=float(load),sweep=sweep,**r)),flush=True))
            inner_total += opt['iterations']
            image = p.image(net,context,load)
            inner = {k:float(v) for k,v in p.metrics(net,context,load).items()}
            d = oldd+c.stagger.omega*(image-oldd)
            q,reaction,mechanical = solve(d,load,q)
            H = committed_history(hp,jnp.asarray(p.m.driver(np.asarray(q))))
            physical = {k:float(v) for k,v in physical_metrics(d,H).items()}
            raw = {k:float(v) for k,v in p.metrics(net,p.context(H),load).items()}
            if not np.isfinite(list(physical.values())+list(inner.values())+list(raw.values())).all():
                raise FloatingPointError('nonfinite refined staggered state')
            row = dict(index=index,load=float(load),load_id=int(a.load_ids[ordinal-1]),sweep=sweep,reaction=reaction,
                damage_min=physical['damage_min'],damage_max=physical['damage_max'],
                fixed_H_image_damage_min=inner['damage_min'],fixed_H_image_damage_max=inner['damage_max'],
                fixed_H_image_AT2_relative=inner['AT2_relative'],coupled_AT2_relative=physical['AT2_relative'],
                raw_current_H_AT2_relative=raw['AT2_relative'],damage_change=float(jnp.max(abs(d-oldd))),
                q_change_over_load=float(jnp.max(abs(q-oldq)))/float(load),
                H_change_relative=float(jnp.max(abs(H-oldh)))/max(float(H.max()),float(oldh.max()),c.outer_stop.scale_floor),
                reaction_change_relative=abs(reaction-oldr)/max(abs(reaction),abs(oldr),c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(dp-d,0))),history_decrease=float(jnp.max(jnp.maximum(hp-H,0))),
                drift_count=int(np.count_nonzero(np.linalg.norm(np.asarray(q).reshape(-1,2)[:p.m.n_material],axis=1)/float(load)>1.08)),
                mechanics=mechanical,parent_sha256=parent_sha,inner_optimization=opt,
                morphology=morphology(d,p.points,p.V,band_width=c.head.background_band_lengths*c.physical.l0,
                    threshold=c.head.background_threshold),physical_diagnostics=physical)
            outer = evaluate_asymmetric(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count,
                negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
            outer['checks']['raw_current_H_AT2'] = dict(value=raw['AT2_relative'],limit=c.reporting.coupled_AT2_relative_limit,
                **{'pass':raw['AT2_relative']<=c.reporting.coupled_AT2_relative_limit})
            passed = all(check['pass'] for check in outer['checks'].values())
            count = count+1 if passed else 0
            outer.update(pass_this_sweep=passed,consecutive=count,
                converged=sweep>=c.outer_stop.min_sweeps and count>=c.outer_stop.consecutive_passes)
            row['outer_convergence'] = outer
            atomic_json(folder/f'sweep_{sweep:04d}.json',row)
            atomic_fields(folder/'pending_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),
                H_used=np.asarray(hu),damage_image=np.asarray(image),d_parent=np.asarray(dp),H_parent=np.asarray(hp),
                d_previous=np.asarray(oldd),q_previous=np.asarray(oldq),H_previous=np.asarray(oldh))
            eqx.tree_serialise_leaves(folder/'pending_network.eqx',net)
            pending = dict(index=index,load=float(load),sweep=sweep,reaction=reaction,
                fields_sha256=sha(folder/'pending_fields.npz'),network_sha256=sha(folder/'pending_network.eqx'))
            atomic_json(folder/'pending.json',pending)
            if sweep in a.checkpoints or outer['converged']:
                checkpoint = folder/f'checkpoint_{sweep:04d}'
                checkpoint.mkdir()
                shutil.copyfile(folder/'pending_fields.npz',checkpoint/'fields.npz')
                shutil.copyfile(folder/'pending_network.eqx',checkpoint/'network.eqx')
                atomic_json(checkpoint/'report.json',dict(**row,fields_sha256=sha(checkpoint/'fields.npz'),
                    network_sha256=sha(checkpoint/'network.eqx')))
            current.update(**pending,status='running',inner_updates=inner_total,
                physical_AT2=physical['AT2_relative'],inner_AT2=inner['AT2_relative'],
                raw_current_H_AT2=raw['AT2_relative'],outer_convergence=outer)
            publish(False,'running')
            print(json.dumps(dict(event='refined_outer',index=index,load=float(load),sweep=sweep,reaction=reaction,
                inner_AT2=inner['AT2_relative'],physical_AT2=physical['AT2_relative'],raw_AT2=raw['AT2_relative'])),flush=True)
            if outer['converged']:
                break
        force = p.m.matrix(np.asarray(d))@np.asarray(q)
        if np.linalg.norm(force[p.m.free])>mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('refined load independent host FEM residual fails')
        np.testing.assert_allclose(force@p.m.load_direction,reaction,rtol=1e-9,atol=1e-6)
        np.testing.assert_array_equal(np.asarray(H),np.maximum(np.asarray(hp),p.m.driver(np.asarray(q))))
        shutil.copyfile(folder/'pending_fields.npz',folder/'fields.npz')
        shutil.copyfile(folder/'pending_network.eqx',folder/'network.eqx')
        report = dict(index=index,load=float(load),load_id=int(a.load_ids[ordinal-1]),sweeps=sweep,
            reaction=reaction,physical_AT2=physical['AT2_relative'],inner_updates=inner_total,
            seconds=time.perf_counter()-started,parent_sha256=parent_sha,state_sha256=sha(folder/'fields.npz'),
            network_sha256=sha(folder/'network.eqx'),outer_convergence=outer,morphology=row['morphology'],
            crack=crack_extent(d,p.points,graph,threshold=.9,tip=[.5,.5],tip_radius=.024,right_edge=1.,tolerance=1e-12),
            **advancement(user_authorized=a.user_provisional,strict_ready=bool(outer['converged']),
                numerically_valid=True,provisional_ancestry=ancestry))
        atomic_json(folder/'result.json',report)
        curve.append(report)
        current.update(status=report['status'],local_strict_ready=report['local_strict_ready'])
        publish(False,'running')
        if not report['load_advanced']:
            publish(True,'convergence_rejected')
            return
        parent_sha = report['state_sha256']
        ancestry = ancestry or not report['local_strict_ready']
    for name,digest in a.pins.items():
        if sha(Path(a.source_root)/name)!=digest:
            raise RuntimeError(f'protected refined source changed: {name}')
    for name,digest in codes.items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise RuntimeError(f'refined numerical source changed: {name}')
    publish(True,'provisional' if ancestry else 'certified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config'](),preflight_only=args.preflight)
