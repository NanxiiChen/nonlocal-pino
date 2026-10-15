"""Fresh bounded outputs without reference fields, clipping or crack-path penalties."""
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
from scipy.sparse import diags, load_npz
from scipy.sparse.linalg import LinearOperator, cg

from examples.fracture2d._legacy.asymmetric_damage_range import evaluate_asymmetric
from examples.fracture2d._legacy.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d._legacy.compact_damage_features import residual_power_loss
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.core_halo_damage import predict_rows
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network, initial_scales, map_damage
from examples.fracture2d._legacy.provisional_damage import advancement, crack_extent, mesh_adjacency
from examples.fracture2d._legacy.range_damage_optimization import damage_diagnostics
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def make_problem(columns, values, matvec, *, head, gain, gc0, gc1, power, floor, volumes, thresholds):
    def image(net, context, load):
        features = jax.lax.stop_gradient(context['features'])
        z = predict_rows(net, features, columns, values, trunk_gain=gain)
        return map_damage(head, z)

    def objective(net, context, load):
        h = jax.lax.stop_gradient(context['H'])
        d = image(net, context, load)
        r = (gc0+2*h)*d+gc1*matvec(d)-2*h
        return residual_power_loss(r, h, power=power, floor=floor)

    def metrics(net, context, load):
        return damage_diagnostics(image(net, context, load), context['H'], matvec,
            gc0=gc0, gc1=gc1, volumes=volumes, thresholds=thresholds)
    return image, objective, metrics


def morphology(d, xy, volumes, *, band_width, threshold):
    d, xy, volumes = np.asarray(d), np.asarray(xy), np.asarray(volumes)
    if (d.ndim != 1 or xy.shape != (len(d), 2) or volumes.shape != d.shape
            or not np.isfinite(d).all() or not np.isfinite(xy).all()
            or not np.isfinite(volumes).all() or np.min(volumes) <= 0
            or not np.isfinite(band_width) or band_width <= 0 or not 0 < threshold < 1):
        raise ValueError('invalid morphology diagnostic arrays/controls')
    outside = abs(xy[:, 1]-.5) > band_width
    if not np.any(outside):
        raise ValueError('morphology diagnostic has no outside-band nodes')
    v = volumes[outside]
    return dict(outside_band_mean=float(v@d[outside]/v.sum()),
        outside_band_fraction_ge_threshold=float(v@(d[outside] >= threshold)/v.sum()),
        outside_band_max=float(d[outside].max()), band_width=band_width,
        threshold=threshold, mask_is_evaluation_only=True)


def run(c):
    a = c.head
    direct_loading = a.mode == 'direct_loading'
    expected_heads = ['sigmoid', 'exp_square'] if direct_loading else ['sigmoid', 'exp_square', 'identity']
    expected_fixed = [52] if direct_loading else [52, 60]
    if (a.mode not in ['direct_loading', 'history_comparison']
            or jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or list(a.heads) != expected_heads
            or list(a.coupled_heads) != ['sigmoid', 'exp_square']
            or list(a.fixed_indices) != expected_fixed or list(a.indices) != list(range(53, 61))
            or a.outer_iterations != 128 or a.inner_iterations != 512
            or a.fixed_iterations != a.lbfgs.maxiter or a.user_provisional is not True
            or c.stagger.omega != .5 or a.lbfgs.residual_limit != c.outer_stop.inner_AT2_relative_limit):
        raise ValueError('bounded morphology requires declared fresh-head and original-model trial')
    sources = {str(a.source52): a.source52_sha256}
    if not direct_loading:
        sources[str(a.source60)] = a.source60_sha256
    if bool(a.warm_root) != bool(a.warm_pins) or (a.warm_root and not direct_loading):
        raise ValueError('same-head warm fits require explicit direct-loading pins')
    for pin in a.warm_pins:
        sources[pin['path']] = pin['sha256']
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError(f'physical source changed: {path}')
    if a.warm_root:
        warm_identity_path = str(Path(a.warm_root)/'identity.json')
        if warm_identity_path not in sources:
            raise ValueError('same-head warm initialization needs pinned ancestry identity')
        warm_identity = json.loads(Path(warm_identity_path).read_text())
        if (not warm_identity['fresh_networks'] or warm_identity['old_weights']
                or warm_identity['direct_targets'] or not warm_identity['no_z0']
                or warm_identity['config']['physical'] != c.physical.to_dict()
                or warm_identity['source'][str(a.source52)] != a.source52_sha256):
            raise ValueError('same-head warm source does not preserve fresh-network/own-H ancestry')
    if sha(c.core.restart.source_report) != c.core.restart.source_report_sha256:
        raise ValueError('certified52 report hash mismatch')
    source_report = json.loads(Path(c.core.restart.source_report).read_text())
    if (source_report['status'] != 'ready' or source_report['index'] != 52
            or source_report['state_sha256'] != a.source52_sha256):
        raise ValueError('bounded restart requires the certified52 physical parent')
    if sha(c.cold.basis_path) != c.cold.basis_sha256:
        raise ValueError('compact basis hash mismatch')
    sources[str(c.cold.basis_path)] = c.cold.basis_sha256
    sources[str(c.core.restart.source_report)] = sha(c.core.restart.source_report)
    m, operator, geometry = load_geometry(c)
    D, B = operator.to_scipy(), load_npz(c.cold.basis_path)
    points, V = m.coordinates[:m.n_material], m.volumes
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    feature, feature_info = make_features(operator, points, gc0=g0, length=c.physical.l0,
        tip=list(c.conditioned.tip), powers=list(c.conditioned.powers), roundoff_factor=c.validation.roundoff_factor)
    feature = eqx.filter_jit(feature)
    matvec = jax.jit(make_padded_matvec(operator)[0])
    volumes = jnp.asarray(V)
    def context(h):
        h = jax.lax.stop_gradient(jnp.asarray(h, jnp.float64))
        return dict(H=h, features=jax.lax.stop_gradient(feature(h)))
    physical = {}
    source_cases = [(52, a.source52)] if direct_loading else [(52, a.source52), (60, a.source60)]
    for i, path in source_cases:
        with np.load(path, allow_pickle=False) as z:
            physical[i] = {k: z[k].copy() for k in z.files}
        if physical[i]['H'].shape != (m.n_material,) or np.min(physical[i]['H']) < 0:
            raise ValueError(f'invalid source history at {i}')
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    names = ['bounded_damage_morphology_trial.py', 'plain_damage_head.py', 'core_halo_damage.py',
        'cached_damage_optimization.py', 'conditioned_damage.py', 'compact_damage_features.py',
        'local_damage_trial.py', 'local_mechanics.py', 'outer_stopping.py', 'asymmetric_damage_range.py']
    code = {name: sha(Path(__file__).with_name(name)) for name in names}
    (root/'code_snapshot').mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), root/'code_snapshot'/name)
    atomic_json(root/'identity.json', dict(config=c.to_dict(), geometry=geometry, source=sources,
        code=code, fresh_networks=True, old_weights=False, direct_targets=False, no_z0=True,
        output_clipping=False, physical_parent_index=52, own_zero_bounded_path=False,
        feature_info=feature_info, morphology_mask_is_evaluation_only=True,
        direct_loading=direct_loading, warm_same_head_only=bool(a.warm_root)))
    def morph(d):
        return morphology(d, points, V, band_width=a.background_band_lengths*c.physical.l0,
            threshold=a.background_threshold)
    def metric(d, h):
        parts = [g0*d, g1*(D@d), 2*h*(d-1)]
        return float(abs(sum(parts)).max()/sum(abs(p).max() for p in parts))
    fixed_reports, paths = [], {}
    def publish(complete, status):
        atomic_json(root/'result.json', dict(complete=complete, status=status,
            fixed_reports=fixed_reports, coupled_paths=paths))
    adaptation_status = 'initial_network_adaptation' if direct_loading else 'fixed_history_training'
    publish(False, adaptation_status)
    models = {}
    VD = diags(V)@D
    skew = VD-VD.T
    if (np.min(V) <= 0 or (skew.nnz and abs(skew.data).max()>1e-10*max(1.,abs(VD.data).max()))):
        raise ValueError('frozen-history oracle requires volume-symmetric positive-volume operator')
    for index in a.fixed_indices:
        h = physical[index]['H']
        folder = root/f'fixed_{index:04d}'
        folder.mkdir()
        oracle = None
        if not direct_loading:
            A = g1*VD+diags(V*(g0+2*h))
            b, diagonal = 2*V*h, A.diagonal()
            if np.min(diagonal) <= 0:
                raise ValueError('nonpositive frozen AT2 diagonal')
            oracle, info = cg(A, b, x0=physical[index]['d'], rtol=c.validation.cg_rtol,
                atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter,
                M=LinearOperator(A.shape, matvec=lambda x: x/diagonal, dtype=np.float64))
            norm, limit = np.linalg.norm(A@oracle-b), max(c.validation.cg_atol,c.validation.cg_rtol*np.linalg.norm(b))
            if info != 0 or norm > limit*(1+1e-3):
                raise RuntimeError(f'evaluation-only fixed-H CG failed: {index}, {info}, {norm}, {limit}')
            atomic_fields(folder/'oracle.npz', d=oracle, H=h)
            atomic_json(folder/'oracle.json', dict(evaluation_only=True, direct_training_targets=False,
                AT2=metric(oracle,h), cg_norm=float(norm), cg_limit=float(limit), morphology=morph(oracle),
                physical_morphology=morph(physical[index]['d']), fields_sha256=sha(folder/'oracle.npz')))
        grad0 = None
        for head in a.heads:
            started = time.perf_counter()
            scales, gain = initial_scales(B,D,h,head=head,initial_damage=a.initial_damage,gc0=g0,gc1=g1)
            # Fixed padded rows are shared with the prior cached full-domain route.
            packed_width = int(np.max(np.diff(B.indptr)))
            bc = np.zeros((B.shape[0], packed_width), np.int32)
            bv = np.zeros(bc.shape, np.float64)
            for row in range(B.shape[0]):
                lo, hi = B.indptr[row:row+2]
                bc[row,:hi-lo] = B.indices[lo:hi]
                bv[row,:hi-lo] = B.data[lo:hi]*scales[B.indices[lo:hi]]
            image, loss, metrics = make_problem(jnp.asarray(bc),jnp.asarray(bv),matvec,head=head,
                gain=gain,gc0=g0,gc1=g1,power=c.core.power,floor=c.conditioned.loss_floor,
                volumes=volumes,thresholds=tuple(c.range.activity_thresholds))
            predict, measure = eqx.filter_jit(image), eqx.filter_jit(metrics)
            net = fresh_network(c,feature_info['input_dimension'],B,head=head)
            ctx, load = context(h), jnp.asarray(index*c.data.load_step,jnp.float64)
            np.testing.assert_allclose(predict(net,ctx,load),a.initial_damage,rtol=1e-12,atol=1e-14)
            grad = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(eqx.filter_grad(loss)(net,ctx,load))])
            if grad0 is None:
                grad0 = grad
            else:
                np.testing.assert_allclose(grad,grad0,rtol=1e-8,atol=1e-10)
            fit = make_cached_lbfgs(net,loss,metrics)[0]
            dst = folder/head
            dst.mkdir()
            np.savez_compressed(dst/'parameterization.npz',column_scale=scales,trunk_gain=gain)
            if a.warm_root:
                warm = Path(a.warm_root)/f'fixed_{index:04d}'/head
                expected_paths = {str(warm/name) for name in ['network.eqx','parameterization.npz','result.json','fields.npz']}
                if not expected_paths.issubset(sources):
                    raise ValueError('missing explicit same-head warm source pins')
                warm_report = json.loads((warm/'result.json').read_text())
                if warm_report['head'] != head or warm_report['index'] != index:
                    raise ValueError('warm network uses a different head/load')
                with np.load(warm/'parameterization.npz') as z:
                    np.testing.assert_array_equal(z['column_scale'],scales)
                    np.testing.assert_array_equal(z['trunk_gain'],gain)
                with np.load(warm/'fields.npz') as z:
                    np.testing.assert_array_equal(z['H'],h)
                    warm_damage = z['d'].copy()
                net = eqx.tree_deserialise_leaves(warm/'network.eqx',net)
                np.testing.assert_allclose(predict(net,ctx,load),warm_damage,rtol=1e-10,atol=1e-12)
            eqx.tree_serialise_leaves(dst/'initial_network.eqx',net)
            net, optimization = fit(net,ctx,load,a.lbfgs.to_dict(),progress=lambda r:print(json.dumps(
                dict(event='adaptation_inner' if direct_loading else 'fixed_inner',index=index,head=head,**r)),flush=True))
            prediction = np.asarray(predict(net,ctx,load))
            atomic_fields(dst/'fields.npz',d=prediction,H=h)
            eqx.tree_serialise_leaves(dst/'network.eqx',net)
            report = dict(index=index,head=head,metrics={k:float(v) for k,v in measure(net,ctx,load).items()},
                morphology=morph(prediction),same_H_direct_damage_L2=None if oracle is None else float(np.sqrt(V@(prediction-oracle)**2/(V@oracle**2))),
                optimization=optimization,seconds=time.perf_counter()-started,
                initial_gradient_parity_inf=float(abs(grad-grad0).max()),fields_sha256=sha(dst/'fields.npz'),
                network_sha256=sha(dst/'network.eqx'),physical_state_promoted=False)
            atomic_json(dst/'result.json',report)
            fixed_reports.append(report)
            if index == 52 and head in a.coupled_heads:
                models[head] = (net,predict,measure,fit)
            publish(False,adaptation_status)
            print(json.dumps(dict(event='adaptation_complete' if direct_loading else 'fixed_complete',index=index,head=head,**report['morphology'],
                AT2=report['metrics']['AT2_relative'],same_H_L2=report['same_H_direct_damage_L2'])),flush=True)
    solve = m.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,maxiter=c.validation.cg_maxiter)
    graph = mesh_adjacency(m.triangles[:m.n_material_cells],population_size=m.n_material)
    field_metrics = eqx.filter_jit(lambda d,h:damage_diagnostics(d,h,matvec,gc0=g0,gc1=g1,
        volumes=volumes,thresholds=tuple(c.range.activity_thresholds)))
    for head in a.coupled_heads:
        net,predict,measure,fit = models[head]
        branch = root/head
        branch.mkdir()
        q,d,H = [jnp.asarray(physical[52][k]) for k in ['q','d','H']]
        parent_sha, ancestry = a.source52_sha256, False
        curve = []
        paths[head] = dict(complete=False,status='running',curve=curve)
        for index in a.indices:
            started = time.perf_counter()
            load = jnp.asarray(index*c.data.load_step,jnp.float64)
            folder = branch/f'state_{index:04d}'
            folder.mkdir()
            dp,hp = d,H
            q,reaction,mechanical = solve(d,load,q)
            H = jnp.maximum(hp,jnp.asarray(m.driver(np.asarray(q))))
            count = 0
            for sweep in range(1,a.outer_iterations+1):
                oldq,oldd,oldh,oldr = q,d,H,reaction
                hu = jax.lax.stop_gradient(H)
                controls = a.lbfgs.to_dict()
                controls['maxiter'] = a.inner_iterations
                net,opt = fit(net,context(hu),load,controls,progress=lambda r:print(json.dumps(
                    dict(event='coupled_inner',head=head,index=index,sweep=sweep,**r)),flush=True))
                image = predict(net,context(hu),load)
                inner = {k:float(v) for k,v in measure(net,context(hu),load).items()}
                d = d+c.stagger.omega*(image-d)
                q,reaction,mechanical = solve(d,load,q)
                H = jnp.maximum(hp,jnp.asarray(m.driver(np.asarray(q))))
                physical_metrics = {k:float(v) for k,v in field_metrics(d,H).items()}
                raw = {k:float(v) for k,v in measure(net,context(H),load).items()}
                if not np.isfinite(list(inner.values())+list(physical_metrics.values())+list(raw.values())).all():
                    raise FloatingPointError('nonfinite bounded coupled state')
                row = dict(index=index,load=float(load),sweep=sweep,reaction=reaction,
                    damage_min=physical_metrics['damage_min'],damage_max=physical_metrics['damage_max'],
                    fixed_H_image_damage_min=inner['damage_min'],fixed_H_image_damage_max=inner['damage_max'],
                    fixed_H_image_AT2_relative=inner['AT2_relative'],coupled_AT2_relative=physical_metrics['AT2_relative'],
                    raw_current_H_AT2_relative=raw['AT2_relative'],damage_change=float(jnp.max(abs(d-oldd))),
                    q_change_over_load=float(jnp.max(abs(q-oldq)))/float(load),
                    H_change_relative=float(jnp.max(abs(H-oldh)))/max(float(H.max()),float(oldh.max()),c.outer_stop.scale_floor),
                    reaction_change_relative=abs(reaction-oldr)/max(abs(reaction),abs(oldr),c.outer_stop.scale_floor),
                    healing=float(jnp.max(jnp.maximum(dp-d,0))),history_decrease=float(jnp.max(jnp.maximum(hp-H,0))),
                    drift_count=int(np.count_nonzero(np.linalg.norm(np.asarray(q).reshape(-1,2)[:m.n_material],axis=1)/float(load)>1.08)),
                    mechanics=mechanical,parent_sha256=parent_sha,inner_optimization=opt,
                    morphology=morph(d),physical_diagnostics=physical_metrics)
                outer = evaluate_asymmetric(row,c.outer_stop,c.reporting,sweep=sweep,previous_consecutive=count,
                    negative_tolerance=c.reporting.negative_range_atol,positive_tolerance=c.reporting.positive_range_atol)
                outer['checks']['raw_current_H_AT2'] = dict(value=raw['AT2_relative'],limit=c.reporting.coupled_AT2_relative_limit,
                    **{'pass':raw['AT2_relative']<=c.reporting.coupled_AT2_relative_limit})
                passed = all(v['pass'] for v in outer['checks'].values())
                count = count+1 if passed else 0
                outer.update(pass_this_sweep=passed,consecutive=count,
                    converged=sweep>=c.outer_stop.min_sweeps and count>=c.outer_stop.consecutive_passes)
                row['outer_convergence'] = outer
                atomic_json(folder/f'sweep_{sweep:04d}.json',row)
                atomic_fields(folder/'pending_fields.npz',q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),
                    H_used=np.asarray(hu),damage_image=np.asarray(image),d_parent=np.asarray(dp),H_parent=np.asarray(hp))
                eqx.tree_serialise_leaves(folder/'pending_network.eqx',net)
                atomic_json(folder/'pending.json',dict(index=index,sweep=sweep,load=float(load),reaction=reaction,
                    fields_sha256=sha(folder/'pending_fields.npz'),network_sha256=sha(folder/'pending_network.eqx')))
                if sweep in a.checkpoints or outer['converged']:
                    checkpoint = folder/f'checkpoint_{sweep:04d}'
                    checkpoint.mkdir()
                    shutil.copyfile(folder/'pending_fields.npz',checkpoint/'fields.npz')
                    shutil.copyfile(folder/'pending_network.eqx',checkpoint/'network.eqx')
                    atomic_json(checkpoint/'report.json',dict(**row,fields_sha256=sha(checkpoint/'fields.npz'),
                        network_sha256=sha(checkpoint/'network.eqx')))
                print(json.dumps(dict(event='bounded_outer',head=head,index=index,sweep=sweep,reaction=reaction,
                    AT2=physical_metrics['AT2_relative'],morphology=row['morphology'])),flush=True)
                if outer['converged']:
                    break
            force = m.matrix(np.asarray(d))@np.asarray(q)
            if np.linalg.norm(force[m.free])>mechanical['convergence_limit']*(1+1e-3):
                raise RuntimeError('independent host FEM force check failed')
            np.testing.assert_allclose(force@m.load_direction,reaction,rtol=1e-9,atol=1e-6)
            np.testing.assert_array_equal(np.asarray(H),np.maximum(np.asarray(hp),m.driver(np.asarray(q))))
            shutil.copyfile(folder/'pending_fields.npz',folder/'fields.npz')
            shutil.copyfile(folder/'pending_network.eqx',folder/'network.eqx')
            report = dict(index=index,load=float(load),reaction=reaction,sweeps=sweep,
                seconds=time.perf_counter()-started,physical_AT2=physical_metrics['AT2_relative'],
                outer_convergence=outer,morphology=morph(d),crack=crack_extent(d,points,graph,threshold=.9,
                    tip=[.5,.5],tip_radius=.024,right_edge=1.,tolerance=1e-12),
                parent_sha256=parent_sha,state_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'),
                **advancement(user_authorized=a.user_provisional,strict_ready=bool(outer['converged']),
                    numerically_valid=True,provisional_ancestry=ancestry))
            atomic_json(folder/'result.json',report)
            curve.append(report)
            parent_sha = report['state_sha256']
            ancestry = ancestry or not report['local_strict_ready']
            publish(False,'coupled_loading')
        paths[head].update(complete=True,status='provisional' if ancestry else 'certified')
        publish(False,'coupled_loading')
    if sources != {p:sha(p) for p in sources} or code != {n:sha(Path(__file__).with_name(n)) for n in code}:
        raise RuntimeError('bounded trial source/code changed during execution')
    publish(True,'complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    run(runpy.run_path(args.config)['get_config']())
